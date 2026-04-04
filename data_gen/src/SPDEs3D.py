import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass(frozen=True)
class Precomp3D:
    mode_numbers_fft: torch.Tensor
    mode_numbers_rfft: torch.Tensor
    lam_rfft: torch.Tensor
    solver_denom: torch.Tensor
    C0: float
    C11: float
    C12: float
    C1: float
    Cmass: float


class SPDE3D:
    def __init__(
        self,
        N,
        dt,
        steps,
        IC=lambda x, y, z: 0,
        dtype=torch.float64,
        seed=0,
        num_tau=256,
        tau_max_multiplier=20.0,
        include_c12=True,
        device=None,
    ):
        self.N = int(N)
        self.dt = float(dt)
        self.steps = int(steps)
        self.IC = IC
        self.dtype = dtype
        self.seed = int(seed)
        self.num_tau = int(num_tau)
        self.tau_max_multiplier = float(tau_max_multiplier)
        self.include_c12 = bool(include_c12)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.pre = None

    @property
    def M(self):
        return 2 * self.N + 1

    @property
    def L(self):
        return 2.0

    @property
    def eps(self):
        return 2.0 / float(self.M)

    def vectorized_3d(self, f, vec1, vec2, vec3):
        v1, v2, v3 = np.meshgrid(vec1, vec2, vec3, indexing="ij")
        if f is None:
            return 0
        if type(f) in {float, int}:
            return f
        return np.vectorize(f)(v1, v2, v3)

    def _torch_dtype(self):
        if self.dtype not in {torch.float32, torch.float64}:
            raise ValueError("dtype must be torch.float32 or torch.float64")
        return self.dtype

    def _float_dtype(self):
        if self._torch_dtype() == torch.float32:
            return np.float32
        return np.float64

    def _fft_mode_numbers(self):
        return np.fft.fftfreq(self.M, d=1.0 / float(self.M)).astype(np.int64)

    def _rfft_mode_numbers(self):
        return np.arange(self.M // 2 + 1, dtype=np.int64)

    def _centered_mode_numbers(self):
        return np.arange(-self.N, self.N + 1, dtype=np.int64)

    def _laplacian_symbol_from_mode_numbers_np(self, kx, ky, kz):
        eps = self.eps
        sx = np.sin(0.5 * math.pi * eps * kx) ** 2
        sy = np.sin(0.5 * math.pi * eps * ky) ** 2
        sz = np.sin(0.5 * math.pi * eps * kz) ** 2
        return (4.0 / (eps * eps)) * (
            sx[:, None, None] + sy[None, :, None] + sz[None, None, :]
        )

    def _laplacian_symbol_from_mode_numbers_torch(self, kx, ky, kz):
        eps = self.eps
        sx = torch.sin(0.5 * math.pi * eps * kx) ** 2
        sy = torch.sin(0.5 * math.pi * eps * ky) ** 2
        sz = torch.sin(0.5 * math.pi * eps * kz) ** 2
        return (4.0 / (eps * eps)) * (
            sx[:, None, None] + sy[None, :, None] + sz[None, None, :]
        )

    def _linear_convolution_3d(self, a, b):
        out_shape = tuple(int(sa + sb - 1) for sa, sb in zip(a.shape, b.shape))
        fa = torch.fft.fftn(a, s=out_shape)
        fb = torch.fft.fftn(b, s=out_shape)
        return torch.fft.ifftn(fa * fb).real

    def _paper_renorm_geometry(self):
        M = self.M
        full_modes = np.arange(-2 * self.N, 2 * self.N + 1, dtype=np.int64)
        mx, my, mz = np.meshgrid(full_modes, full_modes, full_modes, indexing="ij")

        main_mask = (np.abs(mx) <= self.N) & (np.abs(my) <= self.N) & (np.abs(mz) <= self.N)

        shift_x = np.where(mx > self.N, 1, np.where(mx < -self.N, -1, 0))
        shift_y = np.where(my > self.N, 1, np.where(my < -self.N, -1, 0))
        shift_z = np.where(mz > self.N, 1, np.where(mz < -self.N, -1, 0))

        alias_x = mx - M * shift_x
        alias_y = my - M * shift_y
        alias_z = mz - M * shift_z

        centered_modes = self._centered_mode_numbers()
        lam_box = self._laplacian_symbol_from_mode_numbers_np(
            centered_modes, centered_modes, centered_modes
        )
        alias_lam = lam_box[alias_x + self.N, alias_y + self.N, alias_z + self.N]
        return (
            torch.as_tensor(main_mask, dtype=torch.bool, device=self.device),
            torch.as_tensor(alias_lam, dtype=self._torch_dtype(), device=self.device),
        )

    def compute_C0_C1(self):
        centered_modes = torch.as_tensor(
            self._centered_mode_numbers(), dtype=self._torch_dtype(), device=self.device
        )
        lam_box = self._laplacian_symbol_from_mode_numbers_torch(
            centered_modes, centered_modes, centered_modes
        )

        lam_safe = lam_box.clone()
        lam_safe[self.N, self.N, self.N] = torch.inf
        c0 = float((2.0 ** -3) * torch.sum(0.5 / lam_safe).item())

        positive_lam = lam_box[lam_box > 0.0]
        lam_min_pos = float(torch.min(positive_lam).item())
        tau_max = self.tau_max_multiplier / lam_min_pos
        num_tau = max(self.num_tau, 2)
        taus = torch.linspace(
            0.0,
            tau_max,
            num_tau,
            dtype=self._torch_dtype(),
            device=self.device,
        )

        main_mask, alias_lam_full = self._paper_renorm_geometry()
        main_slice = slice(self.N, 3 * self.N + 1)
        integrand11 = torch.empty(num_tau, dtype=self._torch_dtype(), device=self.device)
        integrand12 = torch.empty(num_tau, dtype=self._torch_dtype(), device=self.device)

        positive_mask = lam_box > 0.0
        for i, tau in enumerate(taus):
            P_box = torch.exp(-tau * lam_box)
            V_box = torch.zeros_like(lam_box)
            V_box[positive_mask] = P_box[positive_mask] / (2.0 * lam_box[positive_mask])

            conv_VV = self._linear_convolution_3d(V_box, V_box)

            P_main_full = torch.zeros_like(conv_VV)
            P_main_full[main_slice, main_slice, main_slice] = P_box
            integrand11[i] = torch.sum(P_main_full * conv_VV)

            if self.include_c12:
                P_alias_full = torch.where(
                    main_mask,
                    torch.zeros((), dtype=self._torch_dtype(), device=self.device),
                    torch.exp(-tau * alias_lam_full),
                )
                integrand12[i] = torch.sum(P_alias_full * conv_VV)
            else:
                integrand12[i] = 0.0

        c11 = float((2.0 ** -5) * torch.trapezoid(integrand11, taus).item())
        c12 = float((2.0 ** -5) * torch.trapezoid(integrand12, taus).item())
        c1 = c11 + c12
        cmass = 3.0 * c0 - 9.0 * c1
        return c0, c11, c12, c1, cmass

    def precompute(self):
        fft_modes = torch.as_tensor(
            self._fft_mode_numbers(), dtype=self._torch_dtype(), device=self.device
        )
        rfft_modes = torch.as_tensor(
            self._rfft_mode_numbers(), dtype=self._torch_dtype(), device=self.device
        )

        lam_rfft = self._laplacian_symbol_from_mode_numbers_torch(
            fft_modes, fft_modes, rfft_modes
        ).to(self._torch_dtype())
        solver_denom = (1.0 + self.dt * lam_rfft).to(self._torch_dtype())
        C0, C11, C12, C1, Cmass = self.compute_C0_C1()

        self.pre = Precomp3D(
            mode_numbers_fft=fft_modes,
            mode_numbers_rfft=rfft_modes,
            lam_rfft=lam_rfft,
            solver_denom=solver_denom,
            C0=C0,
            C11=C11,
            C12=C12,
            C1=C1,
            Cmass=Cmass,
        )
        return self.pre

    def _ensure_precomp(self, pre=None):
        if pre is not None:
            return pre
        if self.pre is None:
            return self.precompute()
        return self.pre

    def _noise_real_space(self, generator=None):
        shape = (self.M, self.M, self.M)
        scale = math.sqrt(self.dt) / (self.eps ** 1.5)
        return torch.randn(
            shape,
            generator=generator,
            device=self.device,
            dtype=self._torch_dtype(),
        ) * scale

    def _coerce_field(self, phi):
        if phi is None:
            return torch.zeros((self.M, self.M, self.M), dtype=self._torch_dtype(), device=self.device)
        if isinstance(phi, torch.Tensor):
            field = phi.to(device=self.device, dtype=self._torch_dtype())
        else:
            field = torch.as_tensor(phi, dtype=self._torch_dtype(), device=self.device)
        if tuple(field.shape) != (self.M, self.M, self.M):
            raise ValueError(f"phi0 must have shape {(self.M, self.M, self.M)}, got {tuple(field.shape)}.")
        return field

    def initial_condition(self):
        if isinstance(self.IC, (np.ndarray, torch.Tensor)):
            return self._coerce_field(self.IC)

        grid = np.linspace(-1.0, 1.0, self.M, endpoint=False, dtype=self._float_dtype())
        ic = self.vectorized_3d(self.IC, grid, grid, grid)
        if np.isscalar(ic):
            ic = np.full((self.M, self.M, self.M), ic, dtype=self._float_dtype())
        return self._coerce_field(ic)

    def _semi_implicit_step_from_noise(self, phi, noise, pre=None):
        pre = self._ensure_precomp(pre)
        drift = -(phi ** 3) + pre.Cmass * phi
        rhs = phi + self.dt * drift + noise
        rhs_hat = torch.fft.rfftn(rhs, dim=(-3, -2, -1))
        phi_next_hat = rhs_hat / pre.solver_denom
        phi_next = torch.fft.irfftn(phi_next_hat, s=(self.M, self.M, self.M), dim=(-3, -2, -1))
        return phi_next.to(self._torch_dtype())

    def semi_implicit_step(self, phi, generator=None, pre=None):
        phi = self._coerce_field(phi)
        noise = self._noise_real_space(generator=generator)
        return self._semi_implicit_step_from_noise(phi, noise, pre=pre)

    def simulate(
        self,
        phi0: Optional[torch.Tensor] = None,
        snapshot_every=0,
        burnin=0,
        pre=None,
        generator=None,
        noise=None,
    ):
        pre = self._ensure_precomp(pre)
        if phi0 is None and self.IC is not None:
            phi = self.initial_condition()
        else:
            phi = self._coerce_field(phi0)

        if generator is None and noise is None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.seed)

        if noise is not None:
            noise = torch.as_tensor(noise, dtype=self._torch_dtype(), device=self.device)
            if tuple(noise.shape) != (self.steps, self.M, self.M, self.M):
                raise ValueError(
                    f"noise must have shape {(self.steps, self.M, self.M, self.M)}, got {tuple(noise.shape)}."
                )

        traj = []
        for step in range(self.steps):
            if noise is None:
                phi = self.semi_implicit_step(phi, generator=generator, pre=pre)
            else:
                phi = self._semi_implicit_step_from_noise(phi, noise[step], pre=pre)
            traj.append(phi.clone())

        phi_final = phi
        snapshots = None
        if snapshot_every > 0:
            if traj:
                traj_tensor = torch.stack(traj, dim=0)
            else:
                traj_tensor = torch.empty(
                    (0, self.M, self.M, self.M), dtype=self._torch_dtype(), device=self.device
                )
            start = max(int(burnin), 0)
            traj_post = traj_tensor[start:]
            if traj_post.shape[0] == 0:
                snapshots = traj_post
            else:
                idx = torch.arange(traj_post.shape[0], device=self.device)
                mask = (idx + 1) % int(snapshot_every) == 0
                snapshots = traj_post[mask]

        return phi_final, snapshots

    def structure_factor(self, phi):
        phi = self._coerce_field(phi)
        volume = self.L ** 3
        hat_phi = torch.fft.rfftn(phi, dim=(-3, -2, -1))
        return (hat_phi * torch.conj(hat_phi)).real / volume

    def two_point_correlation(self, phi):
        phi = self._coerce_field(phi)
        power = torch.abs(torch.fft.fftn(phi, dim=(-3, -2, -1))) ** 2
        return torch.fft.ifftn(power, dim=(-3, -2, -1)).real / float(self.M ** 3)

    def to_tcxyz(self, snaps):
        snaps = torch.as_tensor(snaps, dtype=self._torch_dtype(), device=self.device)
        return snaps[:, None, ...]
