import torch
import os

# Reproducibility
torch.manual_seed(0)

# -------------------------
# Parameters
# -------------------------pip uni
B = 64      # batch size (number of samples)
Nx = 8      # number of spatial points
T = 50      # time steps
dt = 1.0 / T

out_dir = "predictions"
os.makedirs(out_dir, exist_ok=True)

# -------------------------
# Helper: Brownian paths
# -------------------------
def brownian_motion(shape, drift=0.0):
    """
    Generate Brownian motion with optional drift.
    shape = (B, Nx, T)
    """
    B, Nx, T = shape
    increments = torch.randn(B, Nx, T) * (dt ** 0.5)
    return drift * torch.arange(T).view(1, 1, T) * dt + increments.cumsum(dim=-1)

# -------------------------
# Case 1: identical distributions (control)
# -------------------------
u = brownian_motion((B, Nx, T))
torch.save(
    {"u_real": u, "u_pred": u.clone()},
    os.path.join(out_dir, "sigw1_same.pt"),
)

# -------------------------
# Case 2: small drift mismatch
# -------------------------
u_real = brownian_motion((B, Nx, T), drift=0.0)
u_pred = brownian_motion((B, Nx, T), drift=0.2)

torch.save(
    {"u_real": u_real, "u_pred": u_pred},
    os.path.join(out_dir, "sigw1_small_drift.pt"),
)

# -------------------------
# Case 3: large drift mismatch
# -------------------------
u_real = brownian_motion((B, Nx, T), drift=0.0)
u_pred = brownian_motion((B, Nx, T), drift=1.0)

torch.save(
    {"u_real": u_real, "u_pred": u_pred},
    os.path.join(out_dir, "sigw1_large_drift.pt"),
)

print("Synthetic SigW1 test data written to predictions/")