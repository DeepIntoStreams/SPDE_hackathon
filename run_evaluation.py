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
    'ACF':       (lambda: AutoCorrelationMetric(max_lag=64),     'ACF difference (lag=64)'),
    'CrossCorr': (lambda: CrossCorrelationMetric(lags=64),       'Cross-correlation difference'),
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

    # Evaluate each file
    results = {}
    for filepath in args.files:
        u_real, u_pred = load_predictions(filepath)
        scores = evaluate(u_real, u_pred, metrics)
        # Remap keys to short names for display
        results[filepath] = {name_map[k]: v for k, v in scores.items()}

    # Print table
    format_table(results, args.metrics)


if __name__ == '__main__':
    main()
