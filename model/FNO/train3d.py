from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from timeit import default_timer

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

CURRENT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = CURRENT_DIR.parents[1]
REPO_ROOT = CURRENT_DIR.parents[2]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from FNO3D import FNO3D  # noqa: E402
from model.utilities import LpLoss, count_params  # noqa: E402

try:
    import wandb
except ImportError:
    wandb = None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    if wandb is not None and wandb.run is not None:
        wandb.log({key: value for key, value in data.items() if value is not None})


def wandb_log_file(path, key=None, as_image=False):
    if wandb is None or wandb.run is None or path is None or not Path(path).exists():
        return
    if as_image and key is not None:
        wandb.log({key: wandb.Image(str(path))})
    wandb.save(str(path))


def finish_wandb():
    if wandb is not None and wandb.run is not None:
        wandb.finish()


class Phi43H5Dataset(Dataset):
    """Lazy Phi43 HDF5 dataset.

    Expected raw HDF5 shapes from gen_phi43_h5.py:
        W, sol: [N, T, X, Y, Z]

    Returned tensors:
        xi: [X, Y, Z, T_sub, 6] with channels [dW, x, y, z, t, u0]
        u:  [X, Y, Z, T_sub]
    """

    def __init__(self, h5_path: str, max_t: int | None = None, sub_t: int = 1):
        self.h5_path = h5_path
        self.max_t = max_t
        self.sub_t = sub_t
        self._h5 = None

        with h5py.File(self.h5_path, "r") as h5:
            self.w_shape = tuple(h5["W"].shape)
            self.sol_shape = tuple(h5["sol"].shape)
            if self.w_shape != self.sol_shape:
                raise ValueError(f"W/sol shape mismatch: W={self.w_shape}, sol={self.sol_shape}")
            if len(self.w_shape) != 5:
                raise ValueError(f"Expected W/sol [N,T,X,Y,Z], got {self.w_shape}")

            _, raw_t, nx, ny, nz = self.w_shape
            stop = raw_t if max_t is None else min(max_t, raw_t)
            self.time_indices = np.arange(0, stop, sub_t, dtype=np.int64)
            self.nx, self.ny, self.nz = nx, ny, nz
            self.nt = len(self.time_indices)

            self.x_grid = self._read_or_default_grid(h5, "X", nx)
            self.y_grid = self._read_or_default_grid(h5, "Y", ny)
            self.z_grid = self._read_or_default_grid(h5, "Z", nz)
            self.t_grid = self._read_or_default_grid(h5, "T", raw_t)[self.time_indices]
            self.grid_features = self._build_grid_features()

    @staticmethod
    def _read_or_default_grid(h5, key, size):
        if key in h5:
            arr = np.asarray(h5[key]).reshape(-1).astype(np.float32)
            if arr.shape[0] == size:
                return arr
        return np.linspace(0.0, 1.0, size, dtype=np.float32)

    def _build_grid_features(self):
        x = torch.from_numpy(self.x_grid)
        y = torch.from_numpy(self.y_grid)
        z = torch.from_numpy(self.z_grid)
        t = torch.from_numpy(self.t_grid.astype(np.float32))
        if float(t.max()) > float(t.min()):
            t = (t - t.min()) / (t.max() - t.min())
        xx, yy, zz, tt = torch.meshgrid(x, y, z, t, indexing="ij")
        return torch.stack((xx, yy, zz, tt), dim=-1).float()

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self):
        return self.w_shape[0]

    def __getitem__(self, index):
        h5 = self._file()
        w_txyz = np.asarray(h5["W"][index, self.time_indices], dtype=np.float32)
        u_txyz = np.asarray(h5["sol"][index, self.time_indices], dtype=np.float32)

        w = torch.from_numpy(np.transpose(w_txyz, (1, 2, 3, 0))).float()
        u = torch.from_numpy(np.transpose(u_txyz, (1, 2, 3, 0))).float()

        dw = torch.zeros_like(w)
        if w.shape[-1] > 1:
            dw[..., 1:] = w[..., 1:] - w[..., :-1]

        u0 = u[..., :1].repeat(1, 1, 1, u.shape[-1])
        xi = torch.cat((dw.unsqueeze(-1), self.grid_features, u0.unsqueeze(-1)), dim=-1)
        return xi, u

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None


def make_splits(dataset, ntrain, nval, ntest):
    total = ntrain + nval + ntest
    if total > len(dataset):
        raise ValueError(f"Requested {total} samples, but dataset only has {len(dataset)}")
    train = Subset(dataset, range(0, ntrain))
    val = Subset(dataset, range(ntrain, ntrain + nval))
    test = Subset(dataset, range(ntrain + nval, total))
    return train, val, test


def relative_l2_loss(myloss, pred, target, ignore_t0=True):
    if ignore_t0 and pred.shape[-1] > 1:
        pred = pred[..., 1:]
        target = target[..., 1:]
    return myloss(pred, target)


def eval_fno3d(model, loader, device, myloss, ignore_t0=True):
    model.eval()
    total = 0.0
    count = len(loader.dataset)
    with torch.no_grad():
        for xi, u in loader:
            xi = xi.to(device)
            u = u.to(device)
            pred = model(xi)
            loss = relative_l2_loss(myloss, pred, u, ignore_t0=ignore_t0)
            total += loss.item()
    return total / max(count, 1)


def measure_inference_time(model, loader, device, warmup_batches=1):
    was_training = model.training
    model.eval()
    total_time = 0.0
    total_batches = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (xi, _) in enumerate(loader):
            xi = xi.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = default_timer()
            _ = model(xi)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = default_timer() - t0
            if batch_idx >= warmup_batches:
                total_time += elapsed
                total_batches += 1
                total_samples += xi.shape[0]

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


def plot_loss_curve(loss_epochs, train_losses, val_losses, save_dir):
    if not train_losses or not val_losses:
        return None
    plt.figure(figsize=(9, 6))
    plt.plot(loss_epochs, train_losses, marker="o", markersize=7, linewidth=3, label="train L2")
    plt.plot(loss_epochs, val_losses, marker="o", markersize=7, linewidth=3, label="val L2")
    plt.title("FNO3D Phi43 L2 Loss", fontsize=22, fontweight="bold")
    plt.xlabel("Epoch", fontsize=18, fontweight="bold")
    plt.ylabel("Relative L2 loss", fontsize=18, fontweight="bold")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=15)
    plt.grid(True, alpha=0.35, linewidth=1.2)
    plt.tight_layout()
    out_path = Path(save_dir) / "loss_curve.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"saved loss curve: {out_path}")
    return out_path


def save_best_checkpoint(model, checkpoint_path, epoch, best_val_loss):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": int(epoch),
            "best_val_loss": float(best_val_loss),
        },
        checkpoint_path,
    )


def load_best_checkpoint(model, checkpoint_path, device):
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        print("best checkpoint not found; using current model for final metrics")
        return None
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    print(f"loaded best checkpoint: {checkpoint_path}")
    if isinstance(checkpoint, dict) and "epoch" in checkpoint and "best_val_loss" in checkpoint:
        print(f"best checkpoint epoch: {checkpoint['epoch']} | best val loss: {checkpoint['best_val_loss']:.6f}")
    return checkpoint


def train_model(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    init_wandb(args)

    dataset = Phi43H5Dataset(args.data_path, max_t=args.max_t, sub_t=args.sub_t)
    train_set, val_set, test_set = make_splits(dataset, args.ntrain, args.nval, args.ntest)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    train_eval_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Dataset raw shape W/sol: {dataset.w_shape}")
    print(f"Subsampled T: {dataset.nt} | spatial: ({dataset.nx}, {dataset.ny}, {dataset.nz})")
    print(f"Split sizes: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    model = FNO3D(
        modes_x=args.modes_x,
        modes_y=args.modes_y,
        modes_z=args.modes_z,
        modes_t=args.modes_t,
        width=args.width,
        n_layers=args.layers,
        dropout=args.dropout,
        padding=args.padding,
    ).to(device)
    print(f"The model has {count_params(model)} parameters")

    myloss = LpLoss(d=3, size_average=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=args.plateau_patience, factor=args.factor, threshold=1e-6, min_lr=1e-7
    )

    best_ckpt_path = save_dir / args.best_ckpt_name
    losses_train = []
    losses_val = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for ep in range(args.epochs):
        model.train()
        for xi, u in train_loader:
            xi = xi.to(device)
            u = u.to(device)
            optimizer.zero_grad()
            pred = model(xi)
            loss = relative_l2_loss(myloss, pred, u, ignore_t0=not args.include_t0)
            loss.backward()
            optimizer.step()

        val_loss = eval_fno3d(model, val_loader, device, myloss, ignore_t0=not args.include_t0)
        test_loss = eval_fno3d(model, test_loader, device, myloss, ignore_t0=not args.include_t0)
        scheduler.step(val_loss)

        improved = val_loss < (best_val_loss - args.delta)
        should_stop = False
        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_best_checkpoint(model, best_ckpt_path, ep, best_val_loss)
        else:
            epochs_without_improvement += 1
            should_stop = epochs_without_improvement >= args.plateau_terminate

        if ep % args.print_every == 0 or ep == args.epochs - 1 or should_stop:
            train_loss = eval_fno3d(model, train_eval_loader, device, myloss, ignore_t0=not args.include_t0)
            losses_train.append(train_loss)
            losses_val.append(val_loss)
            print(
                f"Epoch {ep:04d} | Total Train Loss {train_loss:.6f} | "
                f"Total Val Loss {val_loss:.6f} | Total Test Loss {test_loss:.6f}"
            )
            wandb_log({
                "epoch": ep,
                "loss/train": train_loss,
                "loss/val": val_loss,
                "loss/test": test_loss,
            })

        if should_stop:
            print(f"Early stopping at epoch {ep:04d} | best Val Loss {best_val_loss:.6f} | patience {args.plateau_terminate}")
            break

    loss_epochs = [i * args.print_every for i in range(len(losses_train))]
    loss_curve_path = plot_loss_curve(loss_epochs, losses_train, losses_val, save_dir)
    wandb_log_file(loss_curve_path, key="plots/loss_curve", as_image=True)

    best_checkpoint = load_best_checkpoint(model, best_ckpt_path, device)
    wandb_log_file(best_ckpt_path)

    loss_train = eval_fno3d(model, train_eval_loader, device, myloss, ignore_t0=not args.include_t0)
    loss_val = eval_fno3d(model, val_loader, device, myloss, ignore_t0=not args.include_t0)
    loss_test = eval_fno3d(model, test_loader, device, myloss, ignore_t0=not args.include_t0)
    inference = measure_inference_time(model, test_loader, device)

    print("final loss_train:", loss_train)
    print("final loss_val:", loss_val)
    print("final loss_test:", loss_test)
    print("inference_total_time (test loader, warmup excluded):", inference["total_time"])
    print("inference_avg_batch_time:", inference["avg_batch_time"])
    print("inference_avg_sample_time:", inference["avg_sample_time"])
    print("inference_throughput_samples_per_sec:", inference["throughput"])

    wandb_log({
        "final/loss_train": loss_train,
        "final/loss_val": loss_val,
        "final/loss_test": loss_test,
        "best/epoch": best_checkpoint.get("epoch") if isinstance(best_checkpoint, dict) else None,
        "best/val_loss": best_checkpoint.get("best_val_loss") if isinstance(best_checkpoint, dict) else None,
        **{f"inference/{key}": value for key, value in inference.items()},
    })
    finish_wandb()


def main():
    parser = argparse.ArgumentParser(description="Train plain FNO3D baseline on Phi43 HDF5 data")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--ntrain", type=int, default=840)
    parser.add_argument("--nval", type=int, default=180)
    parser.add_argument("--ntest", type=int, default=180)
    parser.add_argument("--max-t", type=int, default=None)
    parser.add_argument("--sub-t", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--plateau-patience", type=int, default=50)
    parser.add_argument("--plateau-terminate", type=int, default=80)
    parser.add_argument("--delta", type=float, default=2e-4)
    parser.add_argument("--factor", type=float, default=0.5)
    parser.add_argument("--modes-x", type=int, default=8)
    parser.add_argument("--modes-y", type=int, default=8)
    parser.add_argument("--modes-z", type=int, default=8)
    parser.add_argument("--modes-t", type=int, default=12)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--padding", type=int, default=6)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--metric-eval-every", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--include-t0", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="FNO_Phi43")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--best-ckpt-name", type=str, default="FNO_Phi43_best_val.pt")
    args = parser.parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
