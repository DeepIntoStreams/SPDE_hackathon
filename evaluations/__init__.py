"""evaluations — SPDE evaluation utilities.

Public API re-exports from statistics and metrics sub-modules.
"""

from evaluations.statistics import (
    to_numpy,
    cov_torch,
    acf_torch,
    non_stationary_acf_torch,
    cacf_torch,
    rmse,
    mean_abs_diff,
)

from evaluations.metrics import (
    Metric,
    CovarianceMetric,
    AutoCorrelationMetric,
    CrossCorrelationMetric,
    MeanAbsDiffMetric,
    VARMetric,
    ESMetric,
    LpLossMetric,
    HsLossMetric,
    RMSEMetric,
    SpatioTemporalPredictor,
    FVDMetric,
    KVDMetric,
)


def collect_predictions(test_dl, forward_fn, device):
    """Collect all (u_real, u_pred) tensors from a test DataLoader.

    Parameters
    ----------
    test_dl : DataLoader
    forward_fn : callable(batch, device) -> (u_pred, u_real)
        A function that takes one batch (as yielded by the DataLoader) and
        the device, runs inference, and returns (u_pred, u_real).
    device : torch.device

    Returns
    -------
    u_real, u_pred : torch.Tensor (on CPU, concatenated over all batches)
    """
    import torch
    reals, preds = [], []
    with torch.no_grad():
        for batch in test_dl:
            u_pred, u_real = forward_fn(batch, device)
            preds.append(u_pred.cpu())
            reals.append(u_real.cpu())
    return torch.cat(reals), torch.cat(preds)


def evaluate(x_real, x_pred, metrics):
    """Run selected metrics on a pair of tensors.

    Parameters
    ----------
    x_real : torch.Tensor
    x_pred : torch.Tensor
    metrics : list[Metric]
        Any subset of Metric instances to evaluate.

    Returns
    -------
    dict[str, torch.Tensor]
        {metric.name: scalar score} for each metric.
    """
    return {m.name: m.measure(x_real, x_pred) for m in metrics}
