import os
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_gen.src.SPDEs3D import SPDE3D


def _fft_mode_numbers(M):
    return np.fft.fftfreq(M, d=1.0 / float(M)).astype(np.int64)


def _rfft_mode_numbers(M):
    return np.arange(M // 2 + 1, dtype=np.int64)


def _centered_mode_numbers(N):
    return np.arange(-N, N + 1, dtype=np.int64)


def _laplacian_symbol_from_mode_numbers_np(kx, ky, kz, eps):
    sx = np.sin(0.5 * np.pi * eps * kx) ** 2
    sy = np.sin(0.5 * np.pi * eps * ky) ** 2
    sz = np.sin(0.5 * np.pi * eps * kz) ** 2
    return (4.0 / (eps * eps)) * (
        sx[:, None, None] + sy[None, :, None] + sz[None, None, :]
    )


def _linear_convolution_3d(a, b):
    out_shape = tuple(np.array(a.shape) + np.array(b.shape) - 1)
    fa = np.fft.fftn(a, s=out_shape)
    fb = np.fft.fftn(b, s=out_shape)
    return np.fft.ifftn(fa * fb).real


def _paper_renorm_geometry(N, eps):
    M = 2 * N + 1
    full_modes = np.arange(-2 * N, 2 * N + 1, dtype=np.int64)
    mx, my, mz = np.meshgrid(full_modes, full_modes, full_modes, indexing="ij")

    main_mask = (np.abs(mx) <= N) & (np.abs(my) <= N) & (np.abs(mz) <= N)

    shift_x = np.where(mx > N, 1, np.where(mx < -N, -1, 0))
    shift_y = np.where(my > N, 1, np.where(my < -N, -1, 0))
    shift_z = np.where(mz > N, 1, np.where(mz < -N, -1, 0))

    alias_x = mx - M * shift_x
    alias_y = my - M * shift_y
    alias_z = mz - M * shift_z

    centered_modes = _centered_mode_numbers(N)
    lam_box = _laplacian_symbol_from_mode_numbers_np(centered_modes, centered_modes, centered_modes, eps)
    alias_lam = lam_box[alias_x + N, alias_y + N, alias_z + N]
    return main_mask, alias_lam


def _compute_reference_constants(N, dt, num_tau, tau_max_multiplier, include_c12):
    M = 2 * N + 1
    eps = 2.0 / float(M)
    centered_modes = _centered_mode_numbers(N)
    lam_box = _laplacian_symbol_from_mode_numbers_np(centered_modes, centered_modes, centered_modes, eps)

    lam_safe = lam_box.copy()
    lam_safe[N, N, N] = np.inf
    c0 = float((2.0 ** -3) * np.sum(0.5 / lam_safe))

    positive_lam = lam_box[lam_box > 0.0]
    tau_max = float(tau_max_multiplier) / float(np.min(positive_lam))
    taus = np.linspace(0.0, tau_max, max(int(num_tau), 2), dtype=np.float64)

    main_mask, alias_lam_full = _paper_renorm_geometry(N, eps)
    main_slice = slice(N, 3 * N + 1)
    integrand11 = np.empty((taus.shape[0],), dtype=np.float64)
    integrand12 = np.empty((taus.shape[0],), dtype=np.float64)

    for i, tau in enumerate(taus):
        P_box = np.exp(-tau * lam_box)
        V_box = np.zeros_like(lam_box)
        positive_mask = lam_box > 0.0
        V_box[positive_mask] = P_box[positive_mask] / (2.0 * lam_box[positive_mask])

        conv_VV = _linear_convolution_3d(V_box, V_box)

        P_main_full = np.zeros_like(conv_VV)
        P_main_full[main_slice, main_slice, main_slice] = P_box
        integrand11[i] = float(np.sum(P_main_full * conv_VV))

        if include_c12:
            P_alias_full = np.where(main_mask, 0.0, np.exp(-tau * alias_lam_full))
            integrand12[i] = float(np.sum(P_alias_full * conv_VV))
        else:
            integrand12[i] = 0.0

    c11 = float((2.0 ** -5) * np.trapezoid(integrand11, taus))
    c12 = float((2.0 ** -5) * np.trapezoid(integrand12, taus))
    c1 = c11 + c12
    cmass = 3.0 * c0 - 9.0 * c1
    return {
        "C0": c0,
        "C11": c11,
        "C12": c12,
        "C1": c1,
        "Cmass": cmass,
    }


def _reference_precompute(N, dt, num_tau, tau_max_multiplier, include_c12):
    M = 2 * N + 1
    eps = 2.0 / float(M)
    fft_modes = _fft_mode_numbers(M)
    rfft_modes = _rfft_mode_numbers(M)
    lam_rfft = _laplacian_symbol_from_mode_numbers_np(fft_modes, fft_modes, rfft_modes, eps)
    solver_denom = 1.0 + float(dt) * lam_rfft
    return {
        "lam_rfft": lam_rfft,
        "solver_denom": solver_denom,
        **_compute_reference_constants(N, dt, num_tau, tau_max_multiplier, include_c12),
    }


def _reference_step(phi, noise, dt, solver_denom, cmass, M):
    drift = -(phi ** 3) + float(cmass) * phi
    rhs = phi + float(dt) * drift + noise
    rhs_hat = np.fft.rfftn(rhs, axes=(0, 1, 2))
    phi_next_hat = rhs_hat / solver_denom
    phi_next = np.fft.irfftn(phi_next_hat, s=(M, M, M), axes=(0, 1, 2))
    return phi_next.astype(np.float64)


def _reference_structure_factor(phi, volume):
    hat_phi = np.fft.rfftn(phi, axes=(0, 1, 2))
    return (hat_phi * np.conj(hat_phi)).real / volume


def _reference_two_point_correlation(phi):
    M = phi.shape[0]
    power = np.abs(np.fft.fftn(phi, axes=(0, 1, 2))) ** 2
    return np.fft.ifftn(power, axes=(0, 1, 2)).real / float(M ** 3)


def test_phi43_precompute_matches_reference():
    spde = SPDE3D(N=1, dt=0.01, steps=2, num_tau=8, tau_max_multiplier=4.0, seed=7)
    pre = spde.precompute()
    ref = _reference_precompute(
        N=1,
        dt=0.01,
        num_tau=8,
        tau_max_multiplier=4.0,
        include_c12=True,
    )

    np.testing.assert_allclose(pre.lam_rfft.cpu().numpy(), ref["lam_rfft"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        pre.solver_denom.cpu().numpy(), ref["solver_denom"], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        np.array([pre.C0, pre.C11, pre.C12, pre.C1, pre.Cmass]),
        np.array([ref["C0"], ref["C11"], ref["C12"], ref["C1"], ref["Cmass"]]),
        rtol=1e-12,
        atol=1e-12,
    )


def test_phi43_fixed_noise_step_and_diagnostics_match_reference():
    spde = SPDE3D(N=1, dt=0.01, steps=3, num_tau=8, tau_max_multiplier=4.0, seed=0)
    pre = spde.precompute()

    phi0 = np.arange(spde.M ** 3, dtype=np.float64).reshape(spde.M, spde.M, spde.M) / 13.0
    noise = np.linspace(
        -0.25,
        0.25,
        spde.steps * spde.M ** 3,
        dtype=np.float64,
    ).reshape(spde.steps, spde.M, spde.M, spde.M)

    phi_torch = torch.as_tensor(phi0, dtype=torch.float64)
    for step in range(spde.steps):
        phi_torch = spde._semi_implicit_step_from_noise(
            phi_torch, torch.as_tensor(noise[step], dtype=torch.float64), pre=pre
        )

    phi_ref = phi0.copy()
    for step in range(spde.steps):
        phi_ref = _reference_step(phi_ref, noise[step], spde.dt, pre.solver_denom.cpu().numpy(), pre.Cmass, spde.M)

    np.testing.assert_allclose(phi_torch.cpu().numpy(), phi_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        spde.structure_factor(phi_torch).cpu().numpy(),
        _reference_structure_factor(phi_ref, spde.L ** 3),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        spde.two_point_correlation(phi_torch).cpu().numpy(),
        _reference_two_point_correlation(phi_ref),
        rtol=1e-12,
        atol=1e-12,
    )


def test_phi43_simulate_snapshot_shapes():
    spde = SPDE3D(N=1, dt=0.01, steps=4, num_tau=8, tau_max_multiplier=4.0, seed=3)
    pre = spde.precompute()
    phi_final, snapshots = spde.simulate(snapshot_every=2, burnin=1, pre=pre)

    assert phi_final.shape == (spde.M, spde.M, spde.M)
    assert snapshots.shape == (1, spde.M, spde.M, spde.M)
    assert spde.to_tcxyz(snapshots).shape == (1, 1, spde.M, spde.M, spde.M)
