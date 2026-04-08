import pytest
import torch

from data_gen.src.SPDEs3D import SPDE3D

DEFAULT_NUM_BLOCKS = 16
STANDARD_ERROR_EPS = 1e-8
POWER_MEAN_EPS = 1e-30


def _standard_error_of_mean(series: torch.Tensor) -> float:
    """Return the usual standard error for a 1D series."""
    sample_count = int(series.shape[0])
    if sample_count <= 1:
        return 0.0

    sample_size = torch.tensor(float(sample_count), dtype=series.dtype, device=series.device)
    return float(torch.std(series, correction=1) / torch.sqrt(sample_size))


def _blocked_mean_and_standard_error(
    series: torch.Tensor, n_blocks: int = DEFAULT_NUM_BLOCKS
) -> tuple[float, float]:
    """Estimate a mean and blocked standard error for a correlated 1D series."""
    sample_count = int(series.shape[0])
    if sample_count <= 1:
        return float(series.mean()), 0.0

    blocks = n_blocks
    if sample_count < blocks:
        blocks = max(1, sample_count // 2) or 1

    block_size = sample_count // blocks
    if block_size == 0:
        return float(series.mean()), _standard_error_of_mean(series)

    trimmed = series[: block_size * blocks]
    reshaped = trimmed.reshape(blocks, block_size)
    block_means = torch.mean(reshaped, dim=1)
    return float(torch.mean(block_means)), _standard_error_of_mean(block_means)


def _assert_statistically_zero(estimate: float, standard_error: float, n_sigma: float) -> None:
    """Assert that an estimate is consistent with zero at the requested sigma level."""
    assert abs(estimate) <= n_sigma * (standard_error + STANDARD_ERROR_EPS)


def _radial_power_spectrum_equal_time(phi: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Take each snapshot, FFT it, average the power over time, then group modes by
    their integer-valued radius in Fourier space.

    Args:
        phi: Tensor with shape (T, Nx, Ny, Nz)

    Returns:
        dict with keys: shell_ids, P_shell_mean, P_shell_std_over_mean, n_modes
    """
    num_times = int(phi.shape[0])
    size_x = int(phi.shape[1])
    size_y = int(phi.shape[2])
    size_z = int(phi.shape[3])

    mu_t = torch.mean(phi, dim=(1, 2, 3), keepdim=True)
    centered = phi - mu_t

    power_sum = torch.zeros((size_x, size_y, size_z), dtype=centered.dtype, device=centered.device)
    for t in range(num_times):
        ft = torch.fft.fftn(centered[t], dim=(0, 1, 2))
        power_sum = power_sum + (ft.real * ft.real + ft.imag * ft.imag)
    power_by_mode = power_sum / float(num_times)

    kx_idx = (torch.fft.fftfreq(size_x, device=phi.device) * float(size_x)).reshape(size_x, 1, 1)
    ky_idx = (torch.fft.fftfreq(size_y, device=phi.device) * float(size_y)).reshape(1, size_y, 1)
    kz_idx = (torch.fft.fftfreq(size_z, device=phi.device) * float(size_z)).reshape(1, 1, size_z)
    r_idx = torch.sqrt(kx_idx * kx_idx + ky_idx * ky_idx + kz_idx * kz_idx)
    shells = torch.round(r_idx).to(torch.int32)
    max_shell = int(torch.max(shells).item())

    shell_ids = []
    shell_mean_power = []
    shell_relstd = []
    n_modes = []

    for s in range(max_shell + 1):
        mask = shells == s
        if not bool(torch.any(mask).item()):
            continue
        vals = power_by_mode[mask]
        mean_val = float(torch.mean(vals))
        if vals.numel() <= 1:
            relstd_val = 0.0
        else:
            relstd_val = float(torch.std(vals, correction=1) / (torch.mean(vals) + POWER_MEAN_EPS))
        shell_ids.append(s)
        shell_mean_power.append(mean_val)
        shell_relstd.append(relstd_val)
        n_modes.append(int(mask.sum().item()))

    return {
        "shell_ids": torch.tensor(shell_ids, dtype=torch.int32, device=phi.device),
        "P_shell_mean": torch.tensor(shell_mean_power, dtype=phi.dtype, device=phi.device),
        "P_shell_std_over_mean": torch.tensor(shell_relstd, dtype=phi.dtype, device=phi.device),
        "n_modes": torch.tensor(n_modes, dtype=torch.int32, device=phi.device),
    }


def _antisymmetric_statistic(phi: torch.Tensor, tau: int) -> torch.Tensor:
    """For a fixed lag, build the per-time antisymmetric two-time statistic."""
    T = int(phi.shape[0])
    values = []
    for t in range(0, T - tau):
        a = phi[t]
        b = phi[t + tau]
        values.append(torch.mean(a * (b * b) - b * (a * a)))
    return torch.stack(values, dim=0)


@pytest.fixture()
def phi_snaps() -> torch.Tensor:
    """
    Run one Phi^4_3 simulation, throw away a short burn-in, and hand the tests a
    stack of field snapshots with shape (T, M, M, M).
    """
    cutoff = 8
    num_snapshots = 256
    burnin_steps = 64
    eps = 2.0 / float(2 * cutoff + 1)
    dt = 0.01 * eps * eps

    spde = SPDE3D(
        N=cutoff,
        dt=dt,
        steps=burnin_steps + num_snapshots,
        seed=0,
        num_tau=48,
        tau_max_multiplier=12.0,
    )

    pre = spde.precompute()
    _, snaps = spde.simulate(phi0=None, snapshot_every=1, burnin=burnin_steps, pre=pre)
    assert snaps is not None
    assert snaps.shape == (num_snapshots, spde.M, spde.M, spde.M)
    return snaps


def test_z2_symmetry(phi_snaps: torch.Tensor) -> None:
    """
    First check the field does not lean positive or negative overall.
    Then check the third central moment is also near zero, so the sample is not
    noticeably skewed one way.

    We do both checks with blocked standard errors over time, since nearby
    snapshots are correlated and we do not want to pretend they are independent.
    """
    num_times = int(phi_snaps.shape[0])

    mu_global = float(torch.mean(phi_snaps))
    m3_central_global = float(torch.mean((phi_snaps - mu_global) ** 3))

    mu_t = torch.mean(phi_snaps, dim=(1, 2, 3))
    mu_bar, mu_se = _blocked_mean_and_standard_error(
        mu_t, n_blocks=min(DEFAULT_NUM_BLOCKS, max(2, num_times // 2))
    )

    mu_t_center = mu_t.reshape(num_times, 1, 1, 1)
    m3_t = torch.mean((phi_snaps - mu_t_center) ** 3, dim=(1, 2, 3))
    m3_bar, m3_se = _blocked_mean_and_standard_error(
        m3_t, n_blocks=min(DEFAULT_NUM_BLOCKS, max(2, num_times // 2))
    )

    _assert_statistically_zero(mu_bar, mu_se, n_sigma=5.0)
    _assert_statistically_zero(m3_bar, m3_se, n_sigma=5.0)

    assert abs(mu_global) < 5e-2
    assert abs(m3_central_global) < 5e-2


def test_isotropy_equal_time_power(phi_snaps: torch.Tensor) -> None:
    """
    Build an equal-time power spectrum, group Fourier modes into radial shells,
    and check that modes with the same |k| have similar power.

    In plain terms: if the field looks isotropic, direction should not matter
    much once the radius in k-space is fixed.
    """
    spec = _radial_power_spectrum_equal_time(phi_snaps)
    relstd = spec["P_shell_std_over_mean"]
    counts = spec["n_modes"]

    mask = counts >= 24
    if not bool(torch.any(mask).item()):
        mask = counts >= 8

    if bool(torch.any(mask).item()):
        median_relstd = float(torch.median(relstd[mask]))
    else:
        median_relstd = float(torch.median(relstd))

    assert median_relstd < 0.35


def test_time_reversal_antisymmetry(phi_snaps: torch.Tensor) -> None:
    """
    Look at a simple antisymmetric two-time quantity for a few short lags and
    make sure it averages out to zero.

    If the rollout is behaving like a stationary, time-reversal-symmetric
    sample, this forward-vs-backward difference should not keep a real signal.
    """
    num_times = int(phi_snaps.shape[0])
    taus = [tau for tau in [1, 2, 4] if tau < num_times]
    for tau in taus:
        s_t = _antisymmetric_statistic(phi_snaps, tau)
        s_bar, s_se = _blocked_mean_and_standard_error(
            s_t, n_blocks=min(DEFAULT_NUM_BLOCKS, max(2, int(s_t.shape[0]) // 2))
        )
        _assert_statistically_zero(s_bar, s_se, n_sigma=4.0)
