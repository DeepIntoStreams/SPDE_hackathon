from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

CURRENT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = CURRENT_DIR.parents[1]
REPO_ROOT = CURRENT_DIR.parents[2]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from evaluations.metrics import AutoCorrelationMetric, CrossCorrelationMetric, LpLossMetric, SigW1Metric  # noqa: E402  # type: ignore[import-not-found]
from model.NSPDE.neural_spde import NeuralSPDE  # noqa: E402
from model.NSPDE.train3d import (  # noqa: E402
    Phi43H5NSPDEDataset,
    eval_nspde3d,
    measure_inference_time,
)
from model.utilities import LpLoss, count_params  # noqa: E402

try:
    import wandb
except ImportError:
    wandb = None


SUMMARY_KEYS = [
    "loss_test",
    "test_Rel_L2",
    "test_W1_2",
    "test_SigW1",
    "test_AutoCorr",
    "test_CrossCorr",
    "inference_avg_sample_time",
    "inference_throughput",
]


def init_wandb(args):
    if not args.use_wandb:
        return None
    if wandb is None:
        raise ImportError("use_wandb is enabled but wandb is not installed. Run `pip install wandb` first.")
    return wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_name,
        config=vars(args),
    )


def wandb_log(data):
    if wandb is not None and getattr(wandb, "run", None) is not None:
        wandb.log({key: value for key, value in data.items() if value is not None})


def finish_wandb():
    if wandb is not None and getattr(wandb, "run", None) is not None:
        wandb.finish()


def build_model(args, device):
    model = NeuralSPDE(
        dim=3,
        in_channels=1,
        noise_channels=1,
        hidden_channels=args.hidden_channels,
        modes1=args.modes_x,
        modes2=args.modes_y,
        modes3=args.modes_z,
        modes4=args.modes_t,
        n_iter=args.n_iter,
        solver="fixed_point",
    ).to(device)
    print(f"The model has {count_params(model)} parameters")
    return model


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    print(f"loaded checkpoint: {checkpoint_path}")
    if isinstance(checkpoint, dict) and "epoch" in checkpoint and "best_val_loss" in checkpoint:
        print(f"checkpoint epoch: {checkpoint['epoch']} | best val loss: {checkpoint['best_val_loss']:.6f}")
    return checkpoint


def make_eval_loader(args, data_path):
    dataset = Phi43H5NSPDEDataset(data_path, max_t=args.max_t, sub_t=args.sub_t, sub_x=args.sub_x)
    if args.eval_size is not None:
        if len(dataset) < args.eval_size:
            raise ValueError(f"{data_path} has {len(dataset)} samples, but eval_size={args.eval_size}")
        eval_set = Subset(dataset, range(args.eval_size))
    else:
        eval_set = dataset
    loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return dataset, loader


def squeeze_pred(pred):
    if pred.ndim == 6 and pred.shape[1] == 1:
        return pred[:, 0]
    return pred


def collect_predictions(model, loader, device):
    model.eval()
    reals, preds = [], []
    with torch.no_grad():
        for u0, xi, u in loader:
            u0 = u0.to(device)
            xi = xi.to(device)
            pred = squeeze_pred(model(u0, xi))
            preds.append(pred.detach().cpu())
            reals.append(u.detach().cpu())
    return torch.cat(reals, dim=0), torch.cat(preds, dim=0)


def to_path_tensor(u):
    if u.ndim != 5:
        raise ValueError(f"Expected [B,X,Y,Z,T], got {tuple(u.shape)}")
    b, x, y, z, t = u.shape
    return u.permute(0, 4, 1, 2, 3).reshape(b, t, x * y * z)


def block_crosscorr(u_real_path, u_pred_path, block_size=256):
    feature_count = u_real_path.shape[-1]
    weighted_total = 0.0
    total_pairs = 0

    for start in range(0, feature_count, block_size):
        end = min(start + block_size, feature_count)
        block_features = end - start
        if block_features <= 0:
            continue

        real_block = u_real_path[..., start:end]
        pred_block = u_pred_path[..., start:end]
        value = CrossCorrelationMetric().measure(real_block, pred_block)
        pair_count = block_features * (block_features + 1) // 2
        weighted_total += float(value) * pair_count
        total_pairs += pair_count

    if total_pairs == 0:
        raise ValueError("No features available for block CrossCorr.")
    return torch.tensor(weighted_total / total_pairs)


def h1_loss_3d(u_real, u_pred):
    b, nx, ny, nz, nt = u_real.shape
    kx = torch.fft.fftfreq(nx, device=u_real.device).reshape(1, nx, 1, 1, 1)
    ky = torch.fft.fftfreq(ny, device=u_real.device).reshape(1, 1, ny, 1, 1)
    kz = torch.fft.fftfreq(nz, device=u_real.device).reshape(1, 1, 1, nz, 1)
    weight = torch.sqrt(1.0 + kx**2 + ky**2 + kz**2)
    real_ft = torch.fft.fftn(u_real, dim=[1, 2, 3])
    pred_ft = torch.fft.fftn(u_pred, dim=[1, 2, 3])
    diff_norm = torch.linalg.vector_norm(((pred_ft - real_ft) * weight).reshape(b, -1), dim=1)
    real_norm = torch.linalg.vector_norm((real_ft * weight).reshape(b, -1), dim=1)
    return torch.mean(diff_norm / real_norm).real


def evaluate_phi43_metrics(model, loader, device, prefix="test", verbose=True, crosscorr_block_size=256):
    u_real, u_pred = collect_predictions(model, loader, device)
    u_real_path = to_path_tensor(u_real)
    u_pred_path = to_path_tensor(u_pred)
    u_real_sig = u_real_path.permute(0, 2, 1)
    u_pred_sig = u_pred_path.permute(0, 2, 1)

    metric_specs = [
        ("Rel_L2", lambda: LpLossMetric(mode="rel").measure(u_real, u_pred)),
        ("W1_2", lambda: h1_loss_3d(u_real, u_pred)),
        ("SigW1", lambda: SigW1Metric().measure(u_real_sig, u_pred_sig)),
        ("AutoCorr", lambda: AutoCorrelationMetric().measure(u_real_path, u_pred_path)),
        ("CrossCorr", lambda: block_crosscorr(u_real_path, u_pred_path, block_size=crosscorr_block_size)),
    ]

    if verbose:
        print(f"{prefix} metrics:")
    results = {}
    for name, fn in metric_specs:
        try:
            value = fn()
            results[name] = float(value)
            if verbose:
                print(f"{prefix}_{name}: {results[name]:.6f}")
        except (RuntimeError, ValueError, ImportError) as err:
            results[name] = None
            if verbose:
                print(f"{prefix}_{name}: failed ({err})")
    return results


def evaluate_one_dataset(args, model, data_path, dataset_idx, device, myloss):
    dataset, loader = make_eval_loader(args, data_path)
    sample_count = len(loader.dataset)
    print("\n" + "=" * 80)
    print(f"Evaluation dataset {dataset_idx}: {data_path}")
    print(f"Raw W/sol shape: {dataset.w_shape}")
    print(f"Evaluating samples: {sample_count}")
    print(f"Subsampled T: {dataset.nt} | spatial: ({dataset.nx}, {dataset.ny}, {dataset.nz})")

    loss_test = eval_nspde3d(model, loader, device, myloss, ignore_t0=not args.include_t0)
    inference = measure_inference_time(model, loader, device, warmup_batches=args.warmup_batches)
    metrics = evaluate_phi43_metrics(
        model,
        loader,
        device,
        prefix=f"test_dataset_{dataset_idx}",
        crosscorr_block_size=args.crosscorr_block_size,
    )

    result = {
        "dataset": str(data_path),
        "num_samples": sample_count,
        "loss_test": loss_test,
        "test_Rel_L2": metrics.get("Rel_L2"),
        "test_W1_2": metrics.get("W1_2"),
        "test_SigW1": metrics.get("SigW1"),
        "test_AutoCorr": metrics.get("AutoCorr"),
        "test_CrossCorr": metrics.get("CrossCorr"),
        "inference_avg_sample_time": inference.get("avg_sample_time"),
        "inference_throughput": inference.get("throughput"),
        "inference_avg_batch_time": inference.get("avg_batch_time"),
        "inference_total_time": inference.get("total_time"),
    }

    print("dataset result:")
    for key in SUMMARY_KEYS:
        value = result.get(key)
        if value is not None:
            print(f"{key}: {value:.6f}")

    wandb_log({
        "dataset_idx": dataset_idx,
        **{f"dataset_{dataset_idx}/{key}": value for key, value in result.items() if isinstance(value, (int, float))},
    })
    dataset.close()
    return result


def print_summary(results):
    print("\n" + "=" * 80)
    print("Independent test dataset summary (mean +/- std):")
    summary = {}
    for key in SUMMARY_KEYS:
        values = np.asarray([row[key] for row in results if row.get(key) is not None], dtype=np.float64)
        if values.size == 0:
            continue
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if values.size > 1 else 0.0
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"] = std
        print(f"{key}: {mean:.6f} +/- {std:.6f}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate NSPDE Phi43 checkpoint on independent HDF5 test datasets")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--eval-data-paths", type=str, nargs="+", required=True)
    parser.add_argument("--eval-size", type=int, default=180)
    parser.add_argument("--max-t", type=int, default=None)
    parser.add_argument("--sub-t", type=int, default=1)
    parser.add_argument("--sub-x", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--crosscorr-block-size", type=int, default=256)
    parser.add_argument("--modes-x", type=int, default=8)
    parser.add_argument("--modes-y", type=int, default=8)
    parser.add_argument("--modes-z", type=int, default=8)
    parser.add_argument("--modes-t", type=int, default=12)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--n-iter", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--include-t0", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="NSPDE_Phi43")
    parser.add_argument("--wandb-name", type=str, default="NSPDE_phi43_eval")
    args = parser.parse_args()

    try:
        init_wandb(args)
        device = torch.device(args.device)
        model = build_model(args, device)
        load_checkpoint(model, args.checkpoint, device)
        model.eval()
        myloss = LpLoss(d=3, size_average=False)

        results = []
        for idx, data_path in enumerate(args.eval_data_paths, start=1):
            results.append(evaluate_one_dataset(args, model, data_path, idx, device, myloss))

        summary = print_summary(results)
        wandb_log({f"summary/{key}": value for key, value in summary.items()})
    finally:
        finish_wandb()


if __name__ == "__main__":
    main()
