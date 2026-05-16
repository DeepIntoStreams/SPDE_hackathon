from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.NORS.models import NORS1D  # noqa: E402
from model.NORS.features import build_mfv_1d, parabolic_graph  # noqa: E402
from model.NORS.utils import LpLoss, count_params, load_spde_1d, make_splits  # noqa: E402
from model.utilities import EarlyStopping  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, optimizer, loss_fn, device, train: bool) -> float:
    model.train(train)
    total = 0.0
    with torch.set_grad_enabled(train):
        for mfv, target in loader:
            mfv = mfv.to(device)
            target = target.to(device)
            pred = model(mfv)
            loss = loss_fn(pred[:, 1:], target[:, 1:])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item()
    return total / len(loader.dataset)


def make_cached_loaders(data, graph, cfg, device):
    U = torch.from_numpy(data["sol"]).float()
    train_idx, val_idx, test_idx = make_splits(len(data["W"]), cfg.ntrain, cfg.nval, cfg.ntest)

    def subset_tensor(tensor, idx):
        return tensor[list(idx)]

    print("[MFV] building full NORS model feature vector")
    mfv = build_mfv_1d(
        graph,
        data["T"],
        data["X"],
        data["W"],
        U0=data["sol"][:, 0],
        device=device,
        batch_size=cfg.mfv_batch_size,
        eps=cfg.rs_eps,
        boundary=cfg.boundary,
        diff=True,
        noise_scale=cfg.get("noise_scale", 0.1),
    ).float()

    train_set = TensorDataset(subset_tensor(mfv, train_idx), subset_tensor(U, train_idx))
    val_set = TensorDataset(subset_tensor(mfv, val_idx), subset_tensor(U, val_idx))
    test_set = TensorDataset(subset_tensor(mfv, test_idx), subset_tensor(U, test_idx))
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    return train_loader, val_loader, test_loader


@hydra.main(version_base=None, config_path="../config", config_name="nors_phi41")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))
    set_seed(int(cfg.seed))
    device = torch.device(cfg.device if cfg.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(cfg.save_dir, exist_ok=True)
    checkpoint_file = str(Path(cfg.save_dir) / cfg.checkpoint_file)

    data = load_spde_1d(cfg.data_path, max_t=cfg.max_t, sub_t=cfg.sub_t, sub_x=cfg.sub_x)
    graph = parabolic_graph(
        data,
        height=cfg.height,
        free_num=cfg.free_num,
        kernel_deg=cfg.kernel_deg,
        noise_deg=cfg.noise_deg,
        deg=cfg.deg_cutoff,
    )
    train_loader, val_loader, test_loader = make_cached_loaders(data, graph, cfg, device)
    print(f"Graph features ({len(graph)}): {list(graph.keys())}")
    print(f"W: {data['W'].shape}, sol: {data['sol'].shape}, X: {data['X'].shape}, T: {data['T'].shape}")

    model = NORS1D(
        num_tree=len(graph),
        modes_x=cfg.modes_x,
        modes_t=cfg.modes_t,
        width=cfg.width,
        layers=cfg.layers,
        padding=cfg.padding,
    ).to(device)
    print(f"Trainable parameters: {count_params(model)}")

    loss_fn = LpLoss(size_average=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=cfg.plateau_patience,
        factor=cfg.scheduler_factor,
        min_lr=cfg.min_lr,
    )
    early_stopping = EarlyStopping(
        patience=cfg.plateau_terminate,
        verbose=False,
        delta=cfg.delta,
        path=checkpoint_file,
    )

    train_time = 0.0
    for epoch in range(1, cfg.epochs + 1):
        tic = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, loss_fn, device, train=True)
        train_time += time.time() - tic
        val_loss = run_epoch(model, val_loader, optimizer, loss_fn, device, train=False)
        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        if epoch == 1 or epoch % cfg.print_every == 0:
            print(
                f"Epoch {epoch:04d} | train={train_loss:.6f} | val={val_loss:.6f} | "
                f"lr={optimizer.param_groups[0]['lr']:.3e} | avg_epoch_time={train_time / epoch:.3f}s"
            )
        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load(checkpoint_file, map_location=device))
    test_loss = run_epoch(model, test_loader, optimizer, loss_fn, device, train=False)
    print(f"loss_test: {test_loss:.6f}")
    print(f"checkpoint: {checkpoint_file}")


if __name__ == "__main__":
    main()
