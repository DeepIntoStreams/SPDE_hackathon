import argparse
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from vape4d import render
except ImportError as exc:
    raise ImportError(
        "vape4d is required for this visualization. Install it in the same environment "
        "where you run this script."
    ) from exc


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = CURRENT_DIR / "data"
DEFAULT_FULL_PATH = DEFAULT_DATA_DIR / "Phi43_xi_N8_steps256_1_full_renorm.h5"
DEFAULT_NONE_PATH = DEFAULT_DATA_DIR / "Phi43_xi_N8_steps256_1_no_renorm.h5"
DEFAULT_OUTPUT_DIR = CURRENT_DIR / "renders"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render Phi43 full-renormalization vs no-renormalization samples with vape4d."
    )
    parser.add_argument("--full-path", type=Path, default=DEFAULT_FULL_PATH)
    parser.add_argument("--none-path", type=Path, default=DEFAULT_NONE_PATH)
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--time-indices", type=int, nargs="+", default=[0, 64, 128, 192, 256])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--cmap", type=str, default="viridis")
    parser.add_argument("--diff-cmap", type=str, default="hot")
    parser.add_argument("--skip-w-check", action="store_true")
    return parser.parse_args()


def read_sample(path, sample_idx):
    with h5py.File(path, "r") as h5:
        sol = np.asarray(h5["sol"][sample_idx], dtype=np.float32)
        w = np.asarray(h5["W"][sample_idx], dtype=np.float32)
        attrs = dict(h5.attrs)
        cmass = float(np.asarray(h5["Cmass"][()])) if "Cmass" in h5 else attrs.get("Cmass", np.nan)
    return sol, w, attrs, cmass


def ensure_txyz(data):
    if data.ndim != 4:
        raise ValueError(f"Expected a 4D sample [T, X, Y, Z], got shape {data.shape}.")
    return data


def prepare_for_vape4d(data_txyz):
    data_txyz = ensure_txyz(data_txyz)
    data_tzyx = np.transpose(data_txyz, (0, 3, 2, 1))
    data_tczyx = data_tzyx[:, np.newaxis, :, :, :]
    return np.ascontiguousarray(data_tczyx, dtype=np.float32)


def two_frame_window(data_tczyx, time_idx):
    if data_tczyx.shape[0] < 2:
        raise ValueError("vape4d.render requires at least two time frames.")
    if time_idx < data_tczyx.shape[0] - 1:
        return data_tczyx[time_idx : time_idx + 2, 0, :, :, :]
    return data_tczyx[-2:, 0, :, :, :]


def render_frame(data_tczyx, time_idx, colormap, alpha, width, height):
    frames = np.ascontiguousarray(two_frame_window(data_tczyx, time_idx), dtype=np.float32)
    return render(
        frames,
        colormap,
        alpha,
        width=width,
        height=height,
    )


def add_panel(ax, image, title):
    ax.imshow(image)
    ax.set_title(title, fontsize=17, fontweight="bold")
    ax.axis("off")


def add_field_panel(ax, field, title, cmap, vmin=None, vmax=None):
    image = ax.imshow(field, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("x", fontsize=12, fontweight="bold")
    ax.set_ylabel("y", fontsize=12, fontweight="bold")
    return image


def save_time_render(
    output_dir,
    time_idx,
    full_vis,
    none_vis,
    diff_vis,
    full_cmass,
    none_cmass,
    args,
):
    cmap = plt.get_cmap(args.cmap)
    diff_cmap = plt.get_cmap(args.diff_cmap)

    full_img = render_frame(full_vis, time_idx, cmap, args.alpha, args.width, args.height)
    none_img = render_frame(none_vis, time_idx, cmap, args.alpha, args.width, args.height)
    diff_img = render_frame(diff_vis, time_idx, diff_cmap, args.alpha, args.width, args.height)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    add_panel(axes[0], full_img, "Full Renormalization")
    add_panel(axes[1], none_img, "No Renormalization")
    add_panel(axes[2], diff_img, "Absolute Difference")

    fig.suptitle(
        f"Phi43 sample {args.sample_idx} | t-index {time_idx} | "
        f"Cmass full={full_cmass:.6g}, none={none_cmass:.6g}",
        fontsize=18,
        fontweight="bold",
    )
    fig.tight_layout()

    output_path = output_dir / f"phi43_full_vs_none_sample{args.sample_idx}_t{time_idx:03d}.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=args.dpi)
    plt.close(fig)
    print(f"Saved {output_path}")


def save_slice_render(output_dir, time_idx, full, none, diff, full_cmass, none_cmass, args):
    z_idx = full.shape[3] // 2
    full_slice = full[time_idx, :, :, z_idx]
    none_slice = none[time_idx, :, :, z_idx]
    diff_slice = diff[time_idx, :, :, z_idx]

    full_proj = np.max(np.abs(full[time_idx]), axis=2)
    none_proj = np.max(np.abs(none[time_idx]), axis=2)
    diff_proj = np.max(diff[time_idx], axis=2)

    field_vmin = min(float(full_slice.min()), float(none_slice.min()))
    field_vmax = max(float(full_slice.max()), float(none_slice.max()))
    proj_vmax = max(float(full_proj.max()), float(none_proj.max()))

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    field_cmap = plt.get_cmap(args.cmap)
    diff_cmap = plt.get_cmap(args.diff_cmap)

    im0 = add_field_panel(
        axes[0, 0], full_slice, f"Full Renorm z={z_idx}", field_cmap, field_vmin, field_vmax
    )
    add_field_panel(axes[0, 1], none_slice, f"No Renorm z={z_idx}", field_cmap, field_vmin, field_vmax)
    im2 = add_field_panel(axes[0, 2], diff_slice, f"|Difference| z={z_idx}", diff_cmap)

    im3 = add_field_panel(axes[1, 0], full_proj, "Full Max Projection", field_cmap, 0.0, proj_vmax)
    add_field_panel(axes[1, 1], none_proj, "No Renorm Max Projection", field_cmap, 0.0, proj_vmax)
    im5 = add_field_panel(axes[1, 2], diff_proj, "|Difference| Max Projection", diff_cmap)

    fig.colorbar(im0, ax=axes[0, :2], shrink=0.78, location="right")
    fig.colorbar(im2, ax=axes[0, 2], shrink=0.78)
    fig.colorbar(im3, ax=axes[1, :2], shrink=0.78, location="right")
    fig.colorbar(im5, ax=axes[1, 2], shrink=0.78)

    fig.suptitle(
        f"Phi43 sample {args.sample_idx} | t-index {time_idx} | "
        f"Cmass full={full_cmass:.6g}, none={none_cmass:.6g}",
        fontsize=18,
        fontweight="bold",
    )
    fig.tight_layout()

    output_path = output_dir / f"phi43_full_vs_none_sample{args.sample_idx}_t{time_idx:03d}_slices.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=args.dpi)
    plt.close(fig)
    print(f"Saved fallback slice render {output_path}")


def save_summary_plot(output_dir, full, none, diff, args):
    full_energy = np.mean(full**2, axis=(1, 2, 3))
    none_energy = np.mean(none**2, axis=(1, 2, 3))
    diff_energy = np.mean(diff**2, axis=(1, 2, 3))
    time = np.arange(full.shape[0])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(time, full_energy, label="Full renorm", linewidth=2.5)
    ax.plot(time, none_energy, label="No renorm", linewidth=2.5)
    ax.plot(time, diff_energy, label="Mean squared difference", linewidth=2.5)
    ax.set_xlabel("Time index", fontsize=14, fontweight="bold")
    ax.set_ylabel("Mean squared field value", fontsize=14, fontweight="bold")
    ax.set_title("Phi43 Full vs No Renormalization Energy", fontsize=16, fontweight="bold")
    ax.legend(prop={"size": 12, "weight": "bold"})
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path = output_dir / f"phi43_full_vs_none_sample{args.sample_idx}_energy.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=args.dpi)
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Phi43 full renormalization vs no renormalization visualization")
    print("=" * 80)
    print("Full path:", args.full_path)
    print("No-renorm path:", args.none_path)
    print("Sample index:", args.sample_idx)

    full, w_full, full_attrs, full_cmass = read_sample(args.full_path, args.sample_idx)
    none, w_none, none_attrs, none_cmass = read_sample(args.none_path, args.sample_idx)

    if full.shape != none.shape:
        raise ValueError(f"Shape mismatch: full {full.shape}, none {none.shape}.")
    if w_full.shape != w_none.shape:
        raise ValueError(f"Noise shape mismatch: full {w_full.shape}, none {w_none.shape}.")
    if not args.skip_w_check and not np.allclose(w_full, w_none):
        raise ValueError("The full and no-renorm files do not share the same W path.")

    max_time_idx = full.shape[0] - 1
    for time_idx in args.time_indices:
        if time_idx < 0 or time_idx > max_time_idx:
            raise ValueError(f"time index {time_idx} is outside [0, {max_time_idx}].")

    diff = np.abs(full - none)
    full_vis = prepare_for_vape4d(full)
    none_vis = prepare_for_vape4d(none)
    diff_vis = prepare_for_vape4d(diff)

    print("Full sol shape:", full.shape)
    print("No-renorm sol shape:", none.shape)
    print("W shape:", w_full.shape)
    print("Full attrs renorm_mode:", full_attrs.get("renorm_mode", "unknown"))
    print("No-renorm attrs renorm_mode:", none_attrs.get("renorm_mode", "unknown"))
    print("Full Cmass:", full_cmass)
    print("No-renorm Cmass:", none_cmass)
    print("Diff min/mean/max:", float(diff.min()), float(diff.mean()), float(diff.max()))
    print("Output dir:", args.output_dir)

    for time_idx in args.time_indices:
        print(f"Rendering t-index {time_idx}...")
        try:
            save_time_render(
                output_dir=args.output_dir,
                time_idx=time_idx,
                full_vis=full_vis,
                none_vis=none_vis,
                diff_vis=diff_vis,
                full_cmass=full_cmass,
                none_cmass=none_cmass,
                args=args,
            )
        except Exception as exc:
            print(f"vape4d rendering failed at t-index {time_idx}: {exc}")
            save_slice_render(
                output_dir=args.output_dir,
                time_idx=time_idx,
                full=full,
                none=none,
                diff=diff,
                full_cmass=full_cmass,
                none_cmass=none_cmass,
                args=args,
            )

    save_summary_plot(args.output_dir, full, none, diff, args)
    print("Done.")


if __name__ == "__main__":
    main()
