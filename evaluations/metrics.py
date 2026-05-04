import torch
from torch import nn
import numpy as np

import warnings
import iisignature 
from scipy import linalg
from sklearn.metrics.pairwise import polynomial_kernel
from abc import ABC, abstractmethod


from evaluations import statistics as stats

'''
Unified evaluation metrics for SPDE benchmarking.

All metrics follow the same interface:
    metric = SomeMetric(config_params...)
    score  = metric.measure(x_real, x_pred)   # -> torch.Tensor (scalar)

Metric List:
- CovarianceMetric
- AutoCorrelationMetric
- CrossCorrelationMetric
- MeanAbsDiffMetric
- VARMetric
- ESMetric
- LpLossMetric
- HsLossMetric
- RMSEMetric
- FVDMetric
- KVDMetric
- SigW1Metric
'''


class Metric(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def measure(self, x_real: torch.Tensor, x_pred: torch.Tensor) -> torch.Tensor:
        """Compare x_real and x_pred, return a scalar distance / score."""
        pass


# ───────────────────── Distribution-statistic metrics ─────────────────────

class CovarianceMetric(Metric):
    """Mean absolute difference between sample covariance matrices."""

    def __init__(self, transform=lambda x: x):
        self.transform = transform

    @property
    def name(self):
        return 'CovMetric'

    def measure(self, x_real, x_pred):
        cov_real = stats.cov_torch(self.transform(x_real))
        cov_pred = stats.cov_torch(self.transform(x_pred))
        return torch.abs(cov_pred - cov_real.to(cov_pred.device)).mean()


class AutoCorrelationMetric(Metric):
    """RMSE between autocorrelation functions."""

    def __init__(self, max_lag=64, stationary=True, dim=(0, 1),
                 symmetric=False, transform=lambda x: x):
        self.transform = transform
        self.max_lag = max_lag
        self.stationary = stationary
        self.dim = dim
        self.symmetric = symmetric

    @property
    def name(self):
        return 'AcfMetric'

    def measure(self, x_real, x_pred):
        t = self.transform
        if self.stationary:
            acf_real = stats.acf_torch(t(x_real), max_lag=self.max_lag, dim=self.dim)
            acf_pred = stats.acf_torch(t(x_pred), max_lag=self.max_lag, dim=self.dim)
        else:
            acf_real = stats.non_stationary_acf_torch(t(x_real), self.symmetric)
            acf_pred = stats.non_stationary_acf_torch(t(x_pred), self.symmetric)
        return (acf_pred - acf_real.to(acf_pred.device)).pow(2).mean().sqrt()


class CrossCorrelationMetric(Metric):
    """Mean absolute difference between cross-correlation tensors."""

    def __init__(self, lags=64, dim=(0, 1), transform=lambda x: x):
        self.transform = transform
        self.lags = lags
        self.dim = dim

    @property
    def name(self):
        return 'CrossCorrMetric'

    def measure(self, x_real, x_pred):
        cc_real = stats.cacf_torch(self.transform(x_real), self.lags, self.dim)
        cc_pred = stats.cacf_torch(self.transform(x_pred), self.lags, self.dim)
        return torch.abs(cc_pred - cc_real.to(cc_pred.device)).mean()


class MeanAbsDiffMetric(Metric):
    """Element-wise mean absolute difference."""

    def __init__(self, transform=lambda x: x):
        self.transform = transform

    @property
    def name(self):
        return 'MeanAbsDiffMetric'

    def measure(self, x_real, x_pred):
        return stats.mean_abs_diff(self.transform(x_real), self.transform(x_pred))


# ───────────────────── Tail-risk metrics ─────────────────────

class VARMetric(Metric):
    """Value-at-Risk metric.

    Computes the alpha-quantile at each spatiotemporal grid point across samples,
    then measures the mean absolute difference between real and predicted quantiles.

    Supports arbitrary input shapes: [B, T, D], [B, Nx, Ny, T], etc.
    The first dimension is always the batch (sample) dimension.
    """

    def __init__(self, alpha=0.05):
        self.alpha = alpha

    @property
    def name(self):
        return 'VARMetric'

    @staticmethod
    def _compute_var(x, alpha):
        B = x.shape[0]
        x_flat = x.reshape(B, -1)
        sorted_arr, _ = torch.sort(x_flat, dim=0)
        idx = int(alpha * B)
        return sorted_arr[idx]

    def measure(self, x_real, x_pred):
        var_real = self._compute_var(x_real, self.alpha)
        var_pred = self._compute_var(x_pred, self.alpha)
        return torch.abs(var_pred - var_real.to(var_pred.device)).mean()


class ESMetric(Metric):
    """Expected Shortfall (CVaR) metric.

    Computes the conditional tail expectation at each spatiotemporal grid point
    across samples, then measures the mean absolute difference.

    Supports arbitrary input shapes: [B, T, D], [B, Nx, Ny, T], etc.
    The first dimension is always the batch (sample) dimension.
    """

    def __init__(self, alpha=0.05):
        self.alpha = alpha

    @property
    def name(self):
        return 'ESMetric'

    @staticmethod
    def _compute_es(x, alpha):
        B = x.shape[0]
        x_flat = x.reshape(B, -1)
        sorted_arr, _ = torch.sort(x_flat, dim=0)
        idx = int(alpha * B)
        return sorted_arr[:idx + 1].mean(dim=0)

    def measure(self, x_real, x_pred):
        es_real = self._compute_es(x_real, self.alpha)
        es_pred = self._compute_es(x_pred, self.alpha)
        return torch.abs(es_pred - es_real.to(es_pred.device)).mean()


# ───────────────────── Pointwise metrics ─────────────────────

class LpLossMetric(Metric):
    """Relative or absolute Lp norm between two fields."""

    def __init__(self, d=2, p=2, size_average=True, reduction=True, mode='rel'):
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.mode = mode

    @property
    def name(self):
        return 'LpLossMetric'

    def measure(self, x_real, x_pred):
        num_examples = x_real.size()[0]
        if self.mode == 'abs':
            h = 1.0 / (x_real.size()[1] - 1.0)
            all_norms = (h ** (self.d / self.p)) * torch.norm(
                x_pred.view(num_examples, -1) - x_real.view(num_examples, -1), self.p, 1)
            if self.reduction:
                if self.size_average:
                    return torch.mean(all_norms)
                else:
                    return torch.sum(all_norms)
            return all_norms
        else:
            diff_norms = torch.norm(
                x_pred.reshape(num_examples, -1) - x_real.reshape(num_examples, -1), self.p, 1)
            real_norms = torch.norm(x_real.reshape(num_examples, -1), self.p, 1)
            if self.reduction:
                if self.size_average:
                    return torch.mean(diff_norms / real_norms)
                else:
                    return torch.sum(diff_norms / real_norms)
            return diff_norms / real_norms


class HsLossMetric(Metric):
    """Sobolev Hs loss comparing numerical derivatives in Fourier space."""

    def __init__(self, d=2, p=2, k=1, a=None, group=False,
                 size_average=True, reduction=True):
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average
        if a is None:
            a = [1,] * k
        self.a = a

    @property
    def name(self):
        return 'HsLossMetric'

    def _rel(self, x_real, x_pred):
        num_examples = x_real.size()[0]
        diff_norms = torch.norm(
            x_pred.reshape(num_examples, -1) - x_real.reshape(num_examples, -1), self.p, 1)
        real_norms = torch.norm(x_real.reshape(num_examples, -1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / real_norms)
            else:
                return torch.sum(diff_norms / real_norms)
        return diff_norms / real_norms

    def measure(self, x_real, x_pred):
        nx = x_real.size()[1]
        ny = x_real.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x_real = x_real.view(x_real.shape[0], nx, ny, -1)
        x_pred = x_pred.view(x_pred.shape[0], nx, ny, -1)

        k_x = torch.cat((torch.arange(start=0, end=nx // 2, step=1),
                          torch.arange(start=-nx // 2, end=0, step=1)), 0).reshape(nx, 1).repeat(1, ny)
        k_y = torch.cat((torch.arange(start=0, end=ny // 2, step=1),
                          torch.arange(start=-ny // 2, end=0, step=1)), 0).reshape(1, ny).repeat(nx, 1)
        k_x = torch.abs(k_x).reshape(1, nx, ny, 1).to(x_real.device)
        k_y = torch.abs(k_y).reshape(1, nx, ny, 1).to(x_real.device)

        x_real = torch.fft.fftn(x_real, dim=[1, 2])
        x_pred = torch.fft.fftn(x_pred, dim=[1, 2])

        if not balanced:
            weight = 1
            if k >= 1:
                weight += a[0] ** 2 * (k_x ** 2 + k_y ** 2)
            if k >= 2:
                weight += a[1] ** 2 * (k_x ** 4 + 2 * k_x ** 2 * k_y ** 2 + k_y ** 4)
            weight = torch.sqrt(weight)
            loss = self._rel(x_real * weight, x_pred * weight)
        else:
            loss = self._rel(x_real, x_pred)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x ** 2 + k_y ** 2)
                loss += self._rel(x_real * weight, x_pred * weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x ** 4 + 2 * k_x ** 2 * k_y ** 2 + k_y ** 4)
                loss += self._rel(x_real * weight, x_pred * weight)
            loss = loss / (k + 1)

        return loss


class RMSEMetric(Metric):
    """Root Mean Squared Error."""

    @property
    def name(self):
        return 'RMSEMetric'

    def measure(self, x_real, x_pred):
        return (x_pred - x_real).pow(2).mean().sqrt()


# ───────────────────── Feature-space metrics ─────────────────────

class SpatioTemporalPredictor(nn.Module):
    """Conv-based feature extractor for spatiotemporal SPDE data.

    Supports 1D, 2D, and 3D SPDE data:
        dim=2: 1D SPDE [B, T, Nx]           -> Conv2d, in_channels=1
        dim=3: 2D SPDE [B, Nx, Ny, T]       -> Conv3d, in_channels=1
        dim=4: 3D SPDE [B, Nx, Ny, Nz, T]   -> Conv3d, in_channels=Nz
               (Nz is folded into the channel dimension; no Conv4d needed)
    """

    def __init__(self, in_channels, out_size, hidden_channels=64, feature_dim=256, dim=2):
        super().__init__()
        assert dim in (2, 3, 4), "dim must be 2 (1D SPDE), 3 (2D SPDE), or 4 (3D SPDE)"
        self.dim = dim
        self.feature_dim = feature_dim

        if dim == 2:
            Conv = nn.Conv2d
            BN = nn.BatchNorm2d
            Pool = nn.AdaptiveAvgPool2d
        else:
            Conv = nn.Conv3d
            BN = nn.BatchNorm3d
            Pool = nn.AdaptiveAvgPool3d

        self.convs = nn.Sequential(
            Conv(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            BN(hidden_channels),
            nn.ReLU(inplace=True),
            Conv(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1),
            BN(hidden_channels * 2),
            nn.ReLU(inplace=True),
            Conv(hidden_channels * 2, hidden_channels * 4, kernel_size=3, stride=2, padding=1),
            BN(hidden_channels * 4),
            nn.ReLU(inplace=True),
            Pool(1),
        )
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(hidden_channels * 4, feature_dim)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(feature_dim, out_size)

    def _prepare_input(self, x):
        if self.dim == 2:
            return x.unsqueeze(1)
        elif self.dim == 3:
            return x.permute(0, 3, 1, 2).unsqueeze(1)
        else:
            return x.permute(0, 3, 4, 1, 2)

    def forward(self, x):
        x = self._prepare_input(x)
        x = self.convs(x)
        x = self.flatten(x)
        x = self.relu(self.linear1(x))
        return self.linear2(x)

    def extract_features(self, x):
        x = self._prepare_input(x)
        x = self.convs(x)
        x = self.flatten(x)
        return self.linear1(x)


class FVDMetric(Metric):
    """Frechet Video Distance.

    Uses SpatioTemporalPredictor features with Frechet distance
    to compare distributions of spatiotemporal SPDE data.
    """

    def __init__(self, model):
        self.model = model

    @property
    def name(self):
        return 'FVDMetric'

    @staticmethod
    def _calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            "Training and test mean vectors have different lengths"
        assert sigma1.shape == sigma2.shape, \
            "Training and test covariances have different dimensions"

        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ("fid calculation produces singular product; "
                   "adding %s to diagonal of cov estimates" % eps)
            warnings.warn(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        return torch.tensor(
            diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

    def measure(self, x_real, x_pred):
        device = x_real.device
        self.model.to(device).eval()
        with torch.no_grad():
            feat_real = self.model.extract_features(x_real).cpu().numpy()
            feat_pred = self.model.extract_features(x_pred).cpu().numpy()

        mu_real = np.mean(feat_real, axis=0)
        sigma_real = np.cov(feat_real, rowvar=False)
        mu_pred = np.mean(feat_pred, axis=0)
        sigma_pred = np.cov(feat_pred, rowvar=False)
        return self._calculate_frechet_distance(mu_real, sigma_real, mu_pred, sigma_pred)


class KVDMetric(Metric):
    """Kernel Video Distance.

    Uses SpatioTemporalPredictor features with polynomial-kernel MMD
    to compare distributions of spatiotemporal SPDE data.
    """

    def __init__(self, model):
        self.model = model

    @property
    def name(self):
        return 'KVDMetric'

    @staticmethod
    def _sqn(arr):
        flat = np.ravel(arr)
        return flat.dot(flat)

    @staticmethod
    def _mmd2_and_variance(K_XX, K_XY, K_YY, unit_diagonal=False,
                           mmd_est='unbiased', var_at_m=None, ret_var=True):
        m = K_XX.shape[0]
        assert K_XX.shape == (m, m)
        assert K_XY.shape == (m, m)
        assert K_YY.shape == (m, m)
        if var_at_m is None:
            var_at_m = m

        if unit_diagonal:
            diag_X = diag_Y = 1
            sum_diag_X = sum_diag_Y = m
            sum_diag2_X = sum_diag2_Y = m
        else:
            diag_X = np.diagonal(K_XX)
            diag_Y = np.diagonal(K_YY)
            sum_diag_X = diag_X.sum()
            sum_diag_Y = diag_Y.sum()
            sum_diag2_X = KVDMetric._sqn(diag_X)
            sum_diag2_Y = KVDMetric._sqn(diag_Y)

        Kt_XX_sums = K_XX.sum(axis=1) - diag_X
        Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
        K_XY_sums_0 = K_XY.sum(axis=0)
        K_XY_sums_1 = K_XY.sum(axis=1)

        Kt_XX_sum = Kt_XX_sums.sum()
        Kt_YY_sum = Kt_YY_sums.sum()
        K_XY_sum = K_XY_sums_0.sum()

        if mmd_est == 'biased':
            mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                    + (Kt_YY_sum + sum_diag_Y) / (m * m)
                    - 2 * K_XY_sum / (m * m))
        else:
            assert mmd_est in {'unbiased', 'u-statistic'}
            mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m - 1))
            if mmd_est == 'unbiased':
                mmd2 -= 2 * K_XY_sum / (m * m)
            else:
                mmd2 -= 2 * (K_XY_sum - np.trace(K_XY)) / (m * (m - 1))

        if not ret_var:
            return mmd2

        Kt_XX_2_sum = KVDMetric._sqn(K_XX) - sum_diag2_X
        Kt_YY_2_sum = KVDMetric._sqn(K_YY) - sum_diag2_Y
        K_XY_2_sum = KVDMetric._sqn(K_XY)

        dot_XX_XY = Kt_XX_sums.dot(K_XY_sums_1)
        dot_YY_YX = Kt_YY_sums.dot(K_XY_sums_0)

        m1 = m - 1
        m2 = m - 2
        zeta1_est = (
            1 / (m * m1 * m2) * (
                KVDMetric._sqn(Kt_XX_sums) - Kt_XX_2_sum
                + KVDMetric._sqn(Kt_YY_sums) - Kt_YY_2_sum)
            - 1 / (m * m1) ** 2 * (Kt_XX_sum ** 2 + Kt_YY_sum ** 2)
            + 1 / (m * m * m1) * (
                KVDMetric._sqn(K_XY_sums_1) + KVDMetric._sqn(K_XY_sums_0)
                - 2 * K_XY_2_sum)
            - 2 / m ** 4 * K_XY_sum ** 2
            - 2 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
            + 2 / (m ** 3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
        )
        zeta2_est = (
            1 / (m * m1) * (Kt_XX_2_sum + Kt_YY_2_sum)
            - 1 / (m * m1) ** 2 * (Kt_XX_sum ** 2 + Kt_YY_sum ** 2)
            + 2 / (m * m) * K_XY_2_sum
            - 2 / m ** 4 * K_XY_sum ** 2
            - 4 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
            + 4 / (m ** 3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
        )
        var_est = (4 * (var_at_m - 2) / (var_at_m * (var_at_m - 1)) * zeta1_est
                   + 2 / (var_at_m * (var_at_m - 1)) * zeta2_est)

        return mmd2, var_est

    @staticmethod
    def _polynomial_mmd(codes_g, codes_r, degree=3, gamma=None, coef0=1,
                        var_at_m=None, ret_var=True):
        K_XX = polynomial_kernel(codes_g, degree=degree, gamma=gamma, coef0=coef0)
        K_YY = polynomial_kernel(codes_r, degree=degree, gamma=gamma, coef0=coef0)
        K_XY = polynomial_kernel(codes_g, codes_r, degree=degree, gamma=gamma, coef0=coef0)
        return KVDMetric._mmd2_and_variance(K_XX, K_XY, K_YY,
                                            var_at_m=var_at_m, ret_var=ret_var)

    @staticmethod
    def _polynomial_mmd_averages(codes_g, codes_r, n_subsets=50, subset_size=1000,
                                 ret_var=False, **kernel_args):
        m = min(codes_g.shape[0], codes_r.shape[0])
        mmds = np.zeros(n_subsets)
        if ret_var:
            vars = np.zeros(n_subsets)
        choice = np.random.choice
        for i in range(n_subsets):
            g = codes_g[choice(len(codes_g), subset_size, replace=False)]
            r = codes_r[choice(len(codes_r), subset_size, replace=False)]
            o = KVDMetric._polynomial_mmd(g, r, **kernel_args,
                                          var_at_m=m, ret_var=ret_var)
            if ret_var:
                mmds[i], vars[i] = o
            else:
                mmds[i] = o
        return (mmds, vars) if ret_var else torch.tensor(mmds.mean() * 1e3)

    def measure(self, x_real, x_pred):
        device = x_real.device
        self.model.to(device).eval()
        with torch.no_grad():
            feat_real = self.model.extract_features(x_real).cpu().numpy()
            feat_pred = self.model.extract_features(x_pred).cpu().numpy()
        return self._polynomial_mmd_averages(feat_pred, feat_real)


#-Signature based metric using wasserstein based metric: 
class SigW1Metric(Metric):
    """
    Signature Wasserstein-1 distance using signatory.

    Computes the truncated Sig-W1 distance between two path distributions
    by comparing their expected signatures level by level.

    Expects inputs of shape (B, T) or (B, T, D), where B is the batch
    (sample) dimension and T is the time dimension.
    """

    def __init__(self, m=5, time_augment=True):
        self.m = m
        self.time_augment = time_augment

    @property
    def name(self):
        return 'SigW1Metric'

    @staticmethod
    def _time_augment(X):
        """
        Prepend a uniform time channel.
        X: torch.Tensor of shape (B, T, D) -> (B, T, D+1)
        """
        B, T, D = X.shape
        t = torch.linspace(0.0, 1.0, T, device=X.device)
        t = t.view(1, T, 1).expand(B, T, 1)
        return torch.cat([t, X], dim=2)

    @staticmethod
    def _split_levels(sig_flat, d, m):
        """
        Split flat signature into per-level tensors.

        signatory.signature returns a flat tensor of shape
        (B, sum_{k=1}^m d^k)
        """
        levels = []
        idx = 0
        for k in range(1, m + 1):
            size = d ** k
            levels.append(sig_flat[:, idx:idx + size])
            idx += size
        return levels

    @staticmethod
    def _expected_signature(X, m):
        """
        Compute expected signature (mean over batch), per level.

        Parameters
        ----------
        X : torch.Tensor, shape (B, T, D)
        m : int

        Returns
        -------
        list of torch.Tensor
            One tensor per level, averaged over batch
        """
        d = X.shape[2]

        # Compute signature: (B, sigdim)
        
        sigs = iisignature.sig(
            X.detach().cpu().numpy().astype("float32"), m
        )
        sigs = torch.from_numpy(sigs).to(X.device)


        levels = SigW1Metric._split_levels(sigs, d, m)
        return [lvl.mean(dim=0) for lvl in levels]

    @staticmethod
    def _sig_w1(exp_sig_real, exp_sig_pred):
        """
        Sum of L1 norms of level-wise differences.
        """
        return sum(
            torch.sum(torch.abs(r - p))
            for r, p in zip(exp_sig_real, exp_sig_pred)
        )

    def measure(self, x_real, x_pred):
        real = x_real
        pred = x_pred

        # ---- remove channel dim if present: (B,1,Nx,T) → (B,Nx,T)
        if real.ndim == 4 and real.shape[1] == 1:
            real = real.squeeze(1)
            pred = pred.squeeze(1)

        # ---- ensure (B, Nx, T, D)
        if real.ndim == 3:
            real = real.unsqueeze(-1)
            pred = pred.unsqueeze(-1)

        if real.ndim != 4:
            raise ValueError(f"SigW1 received unexpected shape: {real.shape}")

        B, Nx, T, D = real.shape

        # ---- reshape to batch over space
        real = real.reshape(B * Nx, T, D)
        pred = pred.reshape(B * Nx, T, D)

        # ---- clean any stray singleton dims
        while real.ndim > 3:
            real = real.squeeze(-1)
        while pred.ndim > 3:
            pred = pred.squeeze(-1)

        if real.ndim == 2:
            real = real.unsqueeze(-1)
        if pred.ndim == 2:
            pred = pred.unsqueeze(-1)

        # ---- safety check
        assert real.ndim == 3, f"real bad shape: {real.shape}"
        assert pred.ndim == 3, f"pred bad shape: {pred.shape}"

        # ---- time augmentation (if enabled)
        if self.time_augment:
            real = self._time_augment(real)
            pred = self._time_augment(pred)

        # ---- compute signatures (single batched call)
        sig_real = iisignature.sig(
            real.detach().cpu().numpy().astype("float32"), self.m
        )
        sig_pred = iisignature.sig(
            pred.detach().cpu().numpy().astype("float32"), self.m
        )

        sig_real = torch.from_numpy(sig_real)
        sig_pred = torch.from_numpy(sig_pred)

        # ---- reshape back to (B, Nx, feature_dim)
        sig_real = sig_real.view(B, Nx, -1)
        sig_pred = sig_pred.view(B, Nx, -1)

        # ---- spatial average of L1 differences (preserves your definition ✅)
        
        spatial_loss = torch.mean(torch.abs(sig_real - sig_pred))

        return spatial_loss

