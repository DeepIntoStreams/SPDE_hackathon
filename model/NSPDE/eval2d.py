from __future__ import annotations

import argparse
import sys
from pathlib import Path
from timeit import default_timer

import numpy as np
import scipy.io
import torch

CURRENT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = CURRENT_DIR.parents[1]
REPO_ROOT = CURRENT_DIR.parents[2]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from evaluations.metrics import CrossCorrelationMetric, HsLossMetric, LpLossMetric, SigW1Metric  # noqa: E402
from model.NSPDE.neural_spde import NeuralSPDE  # noqa: E402
from model.NSPDE.utilities import dataloader_nspde_2d, eval_nspde  # noqa: E402
from model.utilities import LpLoss, count_params  # noqa: E402


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


def load_phi42_mat(path):
    data = scipy.io.loadmat(path)
    w = np.transpose(data["W"], (0, 2, 3, 1))
    sol = np.transpose(data["sol"], (0, 2, 3, 1))
    return torch.from_numpy(w.astype(np.float32)), torch.from_numpy(sol.astype(np.float32))


def make_loader(data_path, args):
    xi, u = load_phi42_mat(data_path)
    _, loader = dataloader_nspde_2d(
        u=u,
        xi=xi,
        ntrain=1,
        ntest=args.eval_size,
        T=args.T,
        sub_t=args.sub_t,
        sub_x=args.sub_x,
        batch_size=args.batch_size,
    )
    return xi, u, loader


def squeeze_pred(pred):
    if pred.ndim == 5 and pred.shape[1] == 1:
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
    if u.ndim == 5 and u.shape[1] == 1:
        u = u[:, 0]
    if u.ndim != 4:
        raise ValueError(f"Expected [B,X,Y,T], got {tuple(u.shape)}")
    b, x, y, t = u.shape
    return u.permute(0, 3, 1, 2).reshape(b, t, x * y)


def stable_autocorr(x, max_lag=64, eps=1e-8):
    max_lag = min(max_lag, x.shape[1])
    x = x - x.mean((0, 1), keepdim=True)
    var = torch.var(x, unbiased=False, dim=(0, 1))
    acf_list = []
    for lag in range(max_lag):
        y = x[:, lag:] * x[:, :-lag] if lag > 0 else x.pow(2)
        acf_i = torch.mean(y, dim=(0, 1)) / (var + eps)
        acf_list.append(torch.nan_to_num(acf_i, nan=0.0, posinf=0.0, neginf=0.0))
    return torch.stack(acf_list)


def stable_autocorr_metric(u_real_path, u_pred_path, max_lag=64):
    acf_real = stable_autocorr(u_real_path, max_lag=max_lag)
    acf_pred = stable_autocorr(u_pred_path, max_lag=max_lag)
    return (acf_pred - acf_real.to(acf_pred.device)).pow(2).mean().sqrt()


def block_crosscorr(u_real_path, u_pred_path, block_size=128, lags=64):
    feature_count = u_real_path.shape[-1]
    lags = min(lags, u_real_path.shape[1])
    weighted_total = 0.0
    total_pairs = 0

    for start in range(0, feature_count, block_size):
        end = min(start + block_size, feature_count)
        block_features = end - start
        if block_features <= 0:
            continue

        real_block = u_real_path[..., start:end]
        pred_block = u_pred_path[..., start:end]
        value = CrossCorrelationMetric(lags=lags).measure(real_block, pred_block)
        value = torch.nan_to_num(torch.as_tensor(value), nan=0.0, posinf=0.0, neginf=0.0)
        pair_count = block_features * (block_features + 1) // 2
        weighted_total += float(value) * pair_count
        total_pairs += pair_count

    if total_pairs == 0:
        raise ValueError("No features available for block CrossCorr.")
    return torch.tensor(weighted_total / total_pairs)


def evaluate_metrics(model, loader, device, prefix, crosscorr_block_size=128, max_lag=64):
    u_real, u_pred = collect_predictions(model, loader, device)
    u_real_path = to_path_tensor(u_real)
    u_pred_path = to_path_tensor(u_pred)
    u_real_sig = u_real_path.permute(0, 2, 1)
    u_pred_sig = u_pred_path.permute(0, 2, 1)
    metric_specs = [
        ("Rel_L2", lambda: LpLossMetric(mode="rel").measure(u_real, u_pred)),
        ("W1_2", lambda: HsLossMetric(k=1).measure(u_real, u_pred)),
        ("SigW1", lambda: SigW1Metric().measure(u_real_sig, u_pred_sig)),
        ("AutoCorr", lambda: stable_autocorr_metric(u_real_path, u_pred_path, max_lag=max_lag)),
        ("CrossCorr", lambda: block_crosscorr(
            u_real_path,
            u_pred_path,
            block_size=crosscorr_block_size,
            lags=max_lag,
        )),
    ]
    print(f"{prefix} metrics:", flush=True)
    results = {}
    for name, compute_metric in metric_specs:
        try:
            print(f"computing {prefix}_{name}...", flush=True)
            value = compute_metric()
            results[name] = float(value)
            print(f"{prefix}_{name}: {results[name]:.6f}", flush=True)
        except Exception as err:
            results[name] = None
            print(f"{prefix}_{name}: failed ({err})", flush=True)
    return results


def measure_inference_time(model, loader, device, warmup_batches=1):
    was_training = model.training
    model.eval()
    total_time = 0.0
    total_batches = 0
    total_samples = 0
    with torch.no_grad():
        for batch_idx, (u0, xi, _) in enumerate(loader):
            u0 = u0.to(device)
            xi = xi.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = default_timer()
            _ = model(u0, xi)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = default_timer() - t0
            if batch_idx >= warmup_batches:
                total_time += elapsed
                total_batches += 1
                total_samples += u0.shape[0]
    if was_training:
        model.train()
    return {
        "total_time": total_time,
        "num_batches": total_batches,
        "num_samples": total_samples,
        "avg_batch_time": total_time / total_batches if total_batches > 0 else float("nan"),
        "avg_sample_time": total_time / total_samples if total_samples > 0 else float("nan"),
        "throughput": total_samples / total_time if total_time > 0 else float("nan"),
    }


def build_model(args, device):
    model = NeuralSPDE(
        dim=2,
        in_channels=1,
        noise_channels=1,
        hidden_channels=args.hidden_channels,
        n_iter=args.n_iter,
        modes1=args.modes1,
        modes2=args.modes2,
        modes3=args.modes3,
        solver=args.solver,
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


def evaluate_one(args, model, data_path, idx, device, myloss):
    xi, u, loader = make_loader(data_path, args)
    print("\n" + "=" * 80)
    print(f"Evaluation dataset {idx}: {data_path}")
    print(f"W shape: {tuple(xi.shape)}")
    print(f"sol shape: {tuple(u.shape)}")
    print(f"Evaluating samples: {len(loader.dataset)}")
    loss_test = eval_nspde(model, loader, myloss, args.batch_size, device)
    inference = measure_inference_time(model, loader, device, warmup_batches=args.warmup_batches)
    metrics = evaluate_metrics(
        model,
        loader,
        device,
        prefix=f"test_dataset_{idx}",
        crosscorr_block_size=args.crosscorr_block_size,
        max_lag=args.max_lag,
    )
    result = {
        "dataset": str(data_path),
        "num_samples": len(loader.dataset),
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
    return result


def print_summary(results):
    print("\n" + "=" * 80)
    print("Independent test dataset summary (mean +/- std):")
    for key in SUMMARY_KEYS:
        values = np.asarray([row[key] for row in results if row.get(key) is not None], dtype=np.float64)
        if values.size == 0:
            continue
        std = float(values.std(ddof=1)) if values.size > 1 else 0.0
        print(f"{key}: {values.mean():.6f} +/- {std:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate NSPDE Phi42 checkpoint on independent MAT test datasets")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--eval-data-paths", type=str, nargs="+", required=True)
    parser.add_argument("--eval-size", type=int, default=180)
    parser.add_argument("--T", type=int, default=250)
    parser.add_argument("--sub-t", type=int, default=5)
    parser.add_argument("--sub-x", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--max-lag", type=int, default=64)
    parser.add_argument("--crosscorr-block-size", type=int, default=128)
    parser.add_argument("--modes1", type=int, default=16)
    parser.add_argument("--modes2", type=int, default=16)
    parser.add_argument("--modes3", type=int, default=16)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--n-iter", type=int, default=1)
    parser.add_argument("--solver", type=str, default="fixed_point")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = build_model(args, device)
    load_checkpoint(model, args.checkpoint, device)
    myloss = LpLoss(size_average=False)
    results = []
    for idx, data_path in enumerate(args.eval_data_paths, start=1):
        results.append(evaluate_one(args, model, data_path, idx, device, myloss))
    print_summary(results)


if __name__ == "__main__":
    main()
