"""
Training script for GalerkinTransformer2D on 2D SPDE datasets (NSE).

Config: model/config/gt_ns.yaml
Mirrors the structure of model/FNO/train2d.py / train_ns.py.

References:
    Cao, S. (2021). Choose a Transformer: Fourier or Galerkin.
    NeurIPS 2021. https://arxiv.org/abs/2105.14995
    Original code: https://github.com/scaomath/galerkin-transformer (MIT)
"""

import os
import sys
import logging
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from GalerkinTransformer.galerkin_transformer import GalerkinTransformer2D
from utilities import LpLoss, count_params, EarlyStopping

log = logging.getLogger(__name__)


def dataloader_gt_2d(
    u: torch.Tensor,
    xi: torch.Tensor,
    ntrain: int,
    nval: int,
    ntest: int,
    T: int,
    sub_t: int = 1,
    sub_x: int = 1,
    batch_size: int = 20,
    task: str = 'xi',
):
    """
    DataLoader builder for 2D SPDE datasets (NSE).
    Handles both (N, T, Nx, Ny) and (N, T, Nx*Ny) storage formats.
    """
    # Handle flattened spatial dim (as stored in some .mat files)
    if u.ndim == 3:
        side = int(round(u.shape[-1] ** 0.5))
        u  = u.reshape(u.shape[0], u.shape[1], side, side)
        xi = xi.reshape(xi.shape[0], xi.shape[1], side, side)

    xi = xi[:, ::sub_t, ::sub_x, ::sub_x]
    u  = u [:, ::sub_t, ::sub_x, ::sub_x]

    T = min(T, xi.shape[1])
    xi, u = xi[:, :T], u[:, :T]

    xi_train = xi[:ntrain];             u_train = u[:ntrain]
    xi_val   = xi[ntrain:ntrain+nval];  u_val   = u[ntrain:ntrain+nval]
    xi_test  = xi[ntrain+nval:ntrain+nval+ntest]
    u_test   = u [ntrain+nval:ntrain+nval+ntest]

    def make_ds(xi_s, u_s):
        if task == 'xi':
            return torch.utils.data.TensorDataset(xi_s, u_s)
        return torch.utils.data.TensorDataset(u_s[:, 0], xi_s, u_s)

    kw = dict(batch_size=batch_size)
    return (
        torch.utils.data.DataLoader(make_ds(xi_train, u_train), shuffle=True,  **kw),
        torch.utils.data.DataLoader(make_ds(xi_val,   u_val),   shuffle=False, **kw),
        torch.utils.data.DataLoader(make_ds(xi_test,  u_test),  shuffle=False, **kw),
    )


def evaluate(model, loader, loss_fn, device, task):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            if task == 'xi':
                xi_b, u_b = [b.to(device) for b in batch]
                pred = model(xi_b)
            else:
                u0_b, xi_b, u_b = [b.to(device) for b in batch]
                pred = model(xi_b, u0_b)

            B = pred.size(0)
            total += loss_fn(pred.reshape(B, -1), u_b.reshape(B, -1)).item()

    return total / len(loader)


@hydra.main(config_path='../../config', config_name='gt_ns', version_base=None)
def main(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Device: {device}')

    import h5py
    import scipy.io

    ext = os.path.splitext(cfg.data_path)[-1]
    if ext == '.mat':
        raw = scipy.io.loadmat(cfg.data_path)
        u   = torch.from_numpy(raw['sol'].astype('float32'))
        xi  = torch.from_numpy(raw['W'].astype('float32'))
    elif ext in ('.h5', '.hdf5'):
        with h5py.File(cfg.data_path, 'r') as f:
            u  = torch.from_numpy(f['sol'][:].astype('float32'))
            xi = torch.from_numpy(f['W'][:].astype('float32'))
    else:
        raise ValueError(f'Unsupported data format: {ext}')

    train_loader, val_loader, test_loader = dataloader_gt_2d(
        u=u, xi=xi,
        ntrain=cfg.ntrain, nval=cfg.nval, ntest=cfg.ntest,
        T=cfg.T, sub_t=cfg.sub_t, sub_x=cfg.sub_x,
        batch_size=cfg.batch_size, task=cfg.task,
    )

    # Effective dimensions after dataloader subsampling.
    sample_batch = next(iter(train_loader))
    if cfg.task == 'xi':
        sample_xi = sample_batch[0]
    else:
        sample_xi = sample_batch[1]

    effective_T = sample_xi.shape[1]
    effective_dim_x = sample_xi.shape[2]
    effective_dim_y = sample_xi.shape[3]

    log.info(
        f'Effective model grid after subsampling: '
        f'T={effective_T}, dim_x={effective_dim_x}, dim_y={effective_dim_y}'
    )

    model = GalerkinTransformer2D(
        T=effective_T,
        dim_x=effective_dim_x,
        dim_y=effective_dim_y,
        d_model=cfg.d_model,
        n_head=cfg.n_head,
        n_layers=cfg.n_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        task=cfg.task,
    ).to(device)

    log.info(f'Parameters: {count_params(model):,}')

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        patience=cfg.plateau_patience,
        factor=0.5,
        min_lr=1e-6,
    )
    loss_fn = LpLoss(size_average=True)

    best_val = float('inf')
    os.makedirs(cfg.save_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.save_dir, cfg.checkpoint_file)

    early_stopping = EarlyStopping(
        patience=cfg.plateau_terminate,
        delta=cfg.delta,
        path=ckpt_path,
    )

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            if cfg.task == 'xi':
                xi_b, u_b = [b.to(device) for b in batch]
                pred = model(xi_b)
            else:
                u0_b, xi_b, u_b = [b.to(device) for b in batch]
                pred = model(xi_b, u0_b)

            B = pred.size(0)
            loss = loss_fn(pred.reshape(B, -1), u_b.reshape(B, -1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss = evaluate(model, val_loader, loss_fn, device, cfg.task)
        scheduler.step(val_loss)

        if epoch % cfg.print_every == 0:
            log.info(f'Epoch {epoch:4d} | train {train_loss:.4e} | val {val_loss:.4e}')

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            log.info(f'Early stopping at epoch {epoch}')
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_loss = evaluate(model, test_loader, loss_fn, device, cfg.task)
    log.info(f'Test relative L2: {test_loss:.4e}')

    with open(os.path.join(cfg.save_dir, cfg.log_file), 'a') as f:
        f.write(f'{OmegaConf.to_yaml(cfg)}\n')
        f.write(
            f'effective_T: {effective_T}\n'
            f'effective_dim_x: {effective_dim_x}\n'
            f'effective_dim_y: {effective_dim_y}\n'
        )
        f.write(f'test_loss: {test_loss:.6f}\n\n')


if __name__ == '__main__':
    main()