"""Compare trained models on selected evaluation metrics.

Usage examples:

  # Compare two models on default metrics (LpLoss + RMSE)
  python run_evaluation.py \\
      --files predictions/NSPDE_Phi41.pt predictions/FNO_Phi41.pt

  # Pick specific metrics
  python run_evaluation.py \\
      --files predictions/NSPDE_Phi41.pt predictions/FNO_Phi41.pt \\
      --metrics LpLoss RMSE ACF Cov

  # List all available metrics
  python run_evaluation.py --list-metrics

Each .pt file is expected to contain {'u_real': Tensor, 'u_pred': Tensor},
produced by collect_predictions + torch.save (see examples below).

----------------------------------------------------------------------
How to produce prediction files (add to the end of any train script):
----------------------------------------------------------------------

  from evaluations import collect_predictions

  # --- NSPDE / NCDE / NRDE / NCDEFNO ---
  def forward_fn(batch, device):
      u0, xi, u = [x.to(device) for x in batch]
      return model(u0, xi), u

  # --- FNO / WNO ---
  def forward_fn(batch, device):
      xi, u = [x.to(device) for x in batch]
      return model(xi)[..., 0], u

  # --- DeepONet ---
  def forward_fn(batch, device):
      u0, u = [x.to(device) for x in batch]
      return model(u0, grid.to(device)), u

  # --- DLR ---
  def forward_fn(batch, device):
      W, U0, F_Xi, Y = [x.to(device) for x in batch]
      return model(U0, W, F_Xi), Y

  # Collect and save
  u_real, u_pred = collect_predictions(test_loader, forward_fn, device)
  torch.save({'u_real': u_real, 'u_pred': u_pred}, 'predictions/MODEL_EQ.pt')
"""

import argparse
import sys
import os

import torch

from evaluations import evaluate
from evaluations.metrics import (
    LpLossMetric,
    HsLossMetric,
    RMSEMetric,
    CovarianceMetric,
    AutoCorrelationMetric,
    CrossCorrelationMetric,
    MeanAbsDiffMetric,
    VARMetric,
    ESMetric,
)


# ── Metric registry ────────────────────────────────────────────────
# Each entry: short_name -> (factory, description)
METRIC_REGISTRY = {
    'LpLoss':    (lambda: LpLossMetric(mode='rel'),              'Relative L2 loss'),
    'LpLossAbs': (lambda: LpLossMetric(mode='abs'),              'Absolute L2 loss'),
    'HsLoss':    (lambda: HsLossMetric(),                        'Sobolev H1 loss'),
    'RMSE':      (lambda: RMSEMetric(),                          'Root mean squared error'),
    'Cov':       (lambda: CovarianceMetric(),                    'Covariance difference'),
    'ACF':       (lambda: AutoCorrelationMetric(),     'ACF difference (lag=64)'),
    'CrossCorr': (lambda: CrossCorrelationMetric(),       'Cross-correlation difference'),
    'MAD':       (lambda: MeanAbsDiffMetric(),                   'Mean absolute difference'),
    'VaR':       (lambda: VARMetric(alpha=0.05),                 'Value-at-Risk (alpha=0.05)'),
    'ES':        (lambda: ESMetric(alpha=0.05),                  'Expected Shortfall (alpha=0.05)'),
}

DEFAULT_METRICS = ['LpLoss', 'RMSE']


def list_metrics():
    print(f'{"Name":<12} Description')
    print('-' * 50)
    for name, (_, desc) in METRIC_REGISTRY.items():
        print(f'{name:<12} {desc}')


def load_predictions(filepath):
    data = torch.load(filepath, map_location='cpu')
    assert 'u_real' in data and 'u_pred' in data, \
        f"{filepath} must contain 'u_real' and 'u_pred' keys"
    return data['u_real'], data['u_pred']


def format_table(results, metric_names):
    """Pretty-print a comparison table."""
    # Column widths
    name_w = max(len(os.path.basename(f)) for f in results) + 2
    col_w = max(max(len(n) for n in metric_names) + 2, 12)

    # Header
    header = f'{"Model":<{name_w}}'
    for m in metric_names:
        header += f'{m:>{col_w}}'
    print(header)
    print('-' * len(header))

    # Rows
    for filepath, scores in results.items():
        row = f'{os.path.basename(filepath):<{name_w}}'
        for m in metric_names:
            val = scores[m]
            row += f'{val.item():>{col_w}.6f}'
        print(row)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate and compare model predictions.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--files', nargs='+',
                        help='Prediction .pt files to evaluate')
    parser.add_argument('--metrics', nargs='+', default=DEFAULT_METRICS,
                        help=f'Metrics to compute (default: {DEFAULT_METRICS})')
    parser.add_argument('--list-metrics', action='store_true',
                        help='List all available metrics and exit')
    args = parser.parse_args()


    if args.list_metrics:
        list_metrics()
        return

    if not args.files:
        parser.error('--files is required (or use --list-metrics)')

    # Validate metric names
    for name in args.metrics:
        if name not in METRIC_REGISTRY:
            print(f"Unknown metric: '{name}'. Use --list-metrics to see options.")
            sys.exit(1)

    # Build metric instances
    metrics = [METRIC_REGISTRY[name][0]() for name in args.metrics]
    # Map from metric.name -> user-facing short name for display
    name_map = {}
    for short_name, m in zip(args.metrics, metrics):
        name_map[m.name] = short_name

    # Evaluate each file (per-batch; LpLoss/LpLossAbs/HsLoss handled specially)
    results = {}
    for filepath in args.files:
        u_real, u_pred = load_predictions(filepath)
        n_test = int(u_real.shape[0])
        print(f"Evaluating {filepath} on {n_test} test samples...")
        batch_size = 20

        # collect per-batch scores for each metric
        metric_scores = {m.name: [] for m in metrics}
        num_batches = (n_test + batch_size - 1) // batch_size

        for bi in range(num_batches):
            s = bi * batch_size
            e = min(s + batch_size, n_test)
            xr = u_real[s:e]
            xp = u_pred[s:e]
            bsz = int(xr.shape[0])

            for m in metrics:
                # LpLoss (abs or rel) -> same reshape as training
                if isinstance(m, LpLossMetric):
                    xr_flat = xr[..., 1:].reshape(bsz, -1)
                    xp_flat = xp[..., 1:].reshape(bsz, -1)
                    val = m.measure(xr_flat, xp_flat)

                # HsLoss -> use the same slice, then ensure shape (B, Nx, Ny, T)
                elif isinstance(m, HsLossMetric):
                    xr_slice = xr[..., 1:]
                    xp_slice = xp[..., 1:]
                    xr2 = xr_slice.squeeze(1) if xr_slice.ndim == 4 else xr_slice
                    xp2 = xp_slice.squeeze(1) if xp_slice.ndim == 4 else xp_slice
                    xr_hs = xr2.unsqueeze(2) if xr2.ndim == 3 else xr2
                    xp_hs = xp2.unsqueeze(2) if xp2.ndim == 3 else xp2
                    val = m.measure(xr_hs, xp_hs)

                # other metrics: pass batches (remove channel dim if present)
                else:
                    xr_for = xr
                    xp_for = xp.squeeze(1) if (xp.ndim == 4 and xp.shape[1] == 1) else xp
                    val = m.measure(xr_for, xp_for)

                metric_scores[m.name].append(float(val.item()) if isinstance(val, torch.Tensor) else float(val))

        # average per-metric across batches and remap keys
        avg_scores = {mname: torch.tensor(sum(vals) / len(vals)) for mname, vals in metric_scores.items()}
        results[filepath] = {name_map[k]: v for k, v in avg_scores.items()}

    # Print table
    format_table(results, args.metrics)


if __name__ == '__main__':
    main()
