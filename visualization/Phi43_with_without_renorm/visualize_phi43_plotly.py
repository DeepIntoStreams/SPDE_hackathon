import argparse
from pathlib import Path

import h5py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = CURRENT_DIR / "data"
DEFAULT_FULL_PATH = DEFAULT_DATA_DIR / "Phi43_xi_N8_steps256_1_full_renorm.h5"
DEFAULT_NONE_PATH = DEFAULT_DATA_DIR / "Phi43_xi_N8_steps256_1_no_renorm.h5"
DEFAULT_OUTPUT_DIR = CURRENT_DIR / "plotly_renders"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create Plotly 3D visualizations for Phi43 full vs no renormalization."
    )
    parser.add_argument("--full-path", type=Path, default=DEFAULT_FULL_PATH)
    parser.add_argument("--none-path", type=Path, default=DEFAULT_NONE_PATH)
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="+",
        default=None,
        help="Average these sample indices before plotting. If omitted, --sample-idx is used.",
    )
    parser.add_argument("--time-index", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", choices=["volume", "isosurface"], default="isosurface")
    parser.add_argument("--surface-count", type=int, default=6)
    parser.add_argument("--opacity", type=float, default=0.28)
    parser.add_argument("--diff-opacity", type=float, default=0.35)
    parser.add_argument("--full-colorscale", type=str, default="Viridis")
    parser.add_argument("--none-colorscale", type=str, default="Viridis")
    parser.add_argument("--diff-colorscale", type=str, default="Hot")
    parser.add_argument("--percentile-low", type=float, default=55.0)
    parser.add_argument("--percentile-high", type=float, default=99.5)
    parser.add_argument("--diff-percentile-low", type=float, default=65.0)
    parser.add_argument("--diff-percentile-high", type=float, default=99.5)
    parser.add_argument("--relative-eps", type=float, default=1e-6)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Use samples 0..num_samples-1. Ignored if --sample-indices is provided.",
    )
    parser.add_argument(
        "--write-separate",
        action="store_true",
        help="Also write separate no-renorm, full-renorm, and difference HTML files.",
    )
    parser.add_argument("--skip-w-check", action="store_true")
    return parser.parse_args()


def read_sample(path, sample_idx):
    with h5py.File(path, "r") as h5:
        sol = np.asarray(h5["sol"][sample_idx], dtype=np.float32)
        w = np.asarray(h5["W"][sample_idx], dtype=np.float32)
        x = np.asarray(h5["X"], dtype=np.float32)
        y = np.asarray(h5["Y"], dtype=np.float32)
        z = np.asarray(h5["Z"], dtype=np.float32)
        attrs = dict(h5.attrs)
        cmass = float(np.asarray(h5["Cmass"][()])) if "Cmass" in h5 else attrs.get("Cmass", np.nan)
    return sol, w, x, y, z, attrs, cmass


def read_mean_field(path, sample_indices, time_index):
    fields = []
    with h5py.File(path, "r") as h5:
        num_samples, num_times = h5["sol"].shape[:2]
        if time_index < 0 or time_index >= num_times:
            raise ValueError(f"--time-index must be in [0, {num_times - 1}].")
        for sample_idx in sample_indices:
            if sample_idx < 0 or sample_idx >= num_samples:
                raise ValueError(f"sample index {sample_idx} must be in [0, {num_samples - 1}].")
            fields.append(np.asarray(h5["sol"][sample_idx, time_index], dtype=np.float32))
        x = np.asarray(h5["X"], dtype=np.float32)
        y = np.asarray(h5["Y"], dtype=np.float32)
        z = np.asarray(h5["Z"], dtype=np.float32)
        attrs = dict(h5.attrs)
        cmass = float(np.asarray(h5["Cmass"][()])) if "Cmass" in h5 else attrs.get("Cmass", np.nan)
    return np.mean(np.stack(fields, axis=0), axis=0).astype(np.float32), x, y, z, attrs, cmass


def read_mean_abs_difference(full_path, none_path, sample_indices, time_index):
    diff_fields = []
    spatial_means = []
    with h5py.File(full_path, "r") as h5_full, h5py.File(none_path, "r") as h5_none:
        if h5_full["sol"].shape != h5_none["sol"].shape:
            raise ValueError(f"Solution shape mismatch: full {h5_full['sol'].shape}, none {h5_none['sol'].shape}.")
        num_samples, num_times = h5_full["sol"].shape[:2]
        if time_index < 0 or time_index >= num_times:
            raise ValueError(f"--time-index must be in [0, {num_times - 1}].")
        for sample_idx in sample_indices:
            if sample_idx < 0 or sample_idx >= num_samples:
                raise ValueError(f"sample index {sample_idx} must be in [0, {num_samples - 1}].")
            full = np.asarray(h5_full["sol"][sample_idx, time_index], dtype=np.float32)
            none = np.asarray(h5_none["sol"][sample_idx, time_index], dtype=np.float32)
            abs_diff = np.abs(full - none)
            diff_fields.append(abs_diff)
            spatial_means.append(float(abs_diff.mean()))
    mean_abs_diff = np.mean(np.stack(diff_fields, axis=0), axis=0).astype(np.float32)
    return mean_abs_diff, np.asarray(spatial_means, dtype=np.float32)


def read_single_relative_difference(full_path, none_path, sample_idx, time_index, eps):
    with h5py.File(full_path, "r") as h5_full, h5py.File(none_path, "r") as h5_none:
        if h5_full["sol"].shape != h5_none["sol"].shape:
            raise ValueError(f"Solution shape mismatch: full {h5_full['sol'].shape}, none {h5_none['sol'].shape}.")
        num_samples, num_times = h5_full["sol"].shape[:2]
        if sample_idx < 0 or sample_idx >= num_samples:
            raise ValueError(f"--sample-idx must be in [0, {num_samples - 1}].")
        if time_index < 0 or time_index >= num_times:
            raise ValueError(f"--time-index must be in [0, {num_times - 1}].")
        full = np.asarray(h5_full["sol"][sample_idx, time_index], dtype=np.float32)
        none = np.asarray(h5_none["sol"][sample_idx, time_index], dtype=np.float32)
    relative_diff = np.abs(full - none) / (np.abs(full) + float(eps))
    return relative_diff.astype(np.float32)


def read_mean_relative_difference(full_path, none_path, sample_indices, time_index, eps):
    ratio_fields = []
    spatial_means = []
    with h5py.File(full_path, "r") as h5_full, h5py.File(none_path, "r") as h5_none:
        if h5_full["sol"].shape != h5_none["sol"].shape:
            raise ValueError(f"Solution shape mismatch: full {h5_full['sol'].shape}, none {h5_none['sol'].shape}.")
        num_samples, num_times = h5_full["sol"].shape[:2]
        if time_index < 0 or time_index >= num_times:
            raise ValueError(f"--time-index must be in [0, {num_times - 1}].")
        for sample_idx in sample_indices:
            if sample_idx < 0 or sample_idx >= num_samples:
                raise ValueError(f"sample index {sample_idx} must be in [0, {num_samples - 1}].")
            full = np.asarray(h5_full["sol"][sample_idx, time_index], dtype=np.float32)
            none = np.asarray(h5_none["sol"][sample_idx, time_index], dtype=np.float32)
            ratio = np.abs(full - none) / (np.abs(full) + float(eps))
            ratio_fields.append(ratio)
            spatial_means.append(float(ratio.mean()))
    mean_ratio = np.mean(np.stack(ratio_fields, axis=0), axis=0).astype(np.float32)
    return mean_ratio, np.asarray(spatial_means, dtype=np.float32)


def check_matching_noise(full_path, none_path, sample_indices, skip_w_check):
    if skip_w_check:
        return
    with h5py.File(full_path, "r") as h5_full, h5py.File(none_path, "r") as h5_none:
        if h5_full["W"].shape != h5_none["W"].shape:
            raise ValueError(f"Noise shape mismatch: full {h5_full['W'].shape}, none {h5_none['W'].shape}.")
        for sample_idx in sample_indices:
            if not np.allclose(h5_full["W"][sample_idx], h5_none["W"][sample_idx]):
                raise ValueError(f"The full and no-renorm files do not share W path for sample {sample_idx}.")


def mesh_coordinates(x, y, z):
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    return xx.ravel(), yy.ravel(), zz.ravel()


def robust_range(field, low, high, positive_only=False):
    values = np.asarray(field, dtype=np.float32).ravel()
    if positive_only:
        values = values[values > 0]
    if values.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(values, low))
    vmax = float(np.percentile(values, high))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    return vmin, vmax


def make_trace(
    field,
    x_flat,
    y_flat,
    z_flat,
    mode,
    name,
    colorscale,
    opacity,
    surface_count,
    percentile_low,
    percentile_high,
    positive_only=False,
    showscale=True,
):
    isomin, isomax = robust_range(
        np.abs(field) if positive_only else field,
        percentile_low,
        percentile_high,
        positive_only=positive_only,
    )
    values = np.abs(field).ravel() if positive_only else field.ravel()

    common = dict(
        x=x_flat,
        y=y_flat,
        z=z_flat,
        value=values,
        isomin=isomin,
        isomax=isomax,
        opacity=opacity,
        surface_count=surface_count,
        colorscale=colorscale,
        showscale=showscale,
        colorbar=dict(
            title=dict(text=name, font=dict(size=15)),
            tickfont=dict(size=14),
            len=0.42,
            thickness=14,
            y=0.5,
            yanchor="middle",
        ),
    )
    if mode == "volume":
        return go.Volume(name=name, **common)
    return go.Isosurface(name=name, caps=dict(x_show=False, y_show=False, z_show=False), **common)


def figure_layout(title, width, height):
    return dict(
        title=dict(text=title, x=0.5, font=dict(size=22)),
        width=width,
        height=height,
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="cube",
            camera=dict(eye=dict(x=1.45, y=1.45, z=1.15)),
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )


def write_figure(fig, path):
    fig.write_html(str(path.with_suffix(".html")))
    print("Saved", path.with_suffix(".html"))


def make_single_figure(field, x_flat, y_flat, z_flat, args, title, colorscale, positive_only=False):
    trace = make_trace(
        field=field,
        x_flat=x_flat,
        y_flat=y_flat,
        z_flat=z_flat,
        mode=args.mode,
        name=title,
        colorscale=colorscale,
        opacity=args.diff_opacity if positive_only else args.opacity,
        surface_count=args.surface_count,
        percentile_low=args.diff_percentile_low if positive_only else args.percentile_low,
        percentile_high=args.diff_percentile_high if positive_only else args.percentile_high,
        positive_only=positive_only,
    )
    fig = go.Figure(data=[trace])
    fig.update_layout(**figure_layout(title, args.width, args.height))
    return fig


def make_comparison_figure(no_field, full_field, ratio_field, x_flat, y_flat, z_flat, args, title):
    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
    )
    traces = [
        make_trace(
            no_field,
            x_flat,
            y_flat,
            z_flat,
            args.mode,
            "",
            args.none_colorscale,
            args.opacity,
            args.surface_count,
            args.percentile_low,
            args.percentile_high,
            showscale=False,
        ),
        make_trace(
            full_field,
            x_flat,
            y_flat,
            z_flat,
            args.mode,
            "",
            args.full_colorscale,
            args.opacity,
            args.surface_count,
            args.percentile_low,
            args.percentile_high,
            showscale=False,
        ),
        make_trace(
            ratio_field,
            x_flat,
            y_flat,
            z_flat,
            args.mode,
            "Rel. diff",
            args.diff_colorscale,
            args.diff_opacity,
            args.surface_count,
            args.diff_percentile_low,
            args.diff_percentile_high,
            positive_only=True,
            showscale=True,
        ),
    ]
    for idx, trace in enumerate(traces, start=1):
        fig.add_trace(trace, row=1, col=idx)

    axis_style = dict(
        tickfont=dict(size=15),
        tickvals=[-1, -0.5, 0, 0.5, 1],
        ticktext=["-1", "-0.5", "0", "0.5", "1"],
        title=dict(text=""),
    )
    scene_common = dict(
        xaxis=axis_style,
        yaxis=axis_style,
        zaxis=axis_style,
        aspectmode="cube",
        camera=dict(eye=dict(x=1.55, y=1.35, z=1.2)),
    )
    fig.update_layout(
        width=max(1500, args.width * 3 // 2),
        height=args.height,
        margin=dict(l=10, r=10, t=5, b=5),
        scene=dict(**scene_common, domain=dict(x=[0.00, 0.29], y=[0.0, 1.0])),
        scene2=dict(**scene_common, domain=dict(x=[0.34, 0.63], y=[0.0, 1.0])),
        scene3=dict(**scene_common, domain=dict(x=[0.68, 0.97], y=[0.0, 1.0])),
    )
    return fig


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.sample_indices is not None:
        sample_indices = args.sample_indices
    elif args.num_samples is not None:
        sample_indices = list(range(args.num_samples))
    else:
        sample_indices = [args.sample_idx]

    full_field, x, y, z, full_attrs, full_cmass = read_mean_field(
        args.full_path, sample_indices, args.time_index
    )
    none_field, x_none, y_none, z_none, none_attrs, none_cmass = read_mean_field(
        args.none_path, sample_indices, args.time_index
    )

    if full_field.shape != none_field.shape:
        raise ValueError(f"Shape mismatch: full {full_field.shape}, none {none_field.shape}.")
    if not (np.allclose(x, x_none) and np.allclose(y, y_none) and np.allclose(z, z_none)):
        raise ValueError("Grid coordinates do not match between full and no-renorm files.")
    check_matching_noise(args.full_path, args.none_path, sample_indices, args.skip_w_check)

    diff_field, sample_spatial_mean_diffs = read_mean_abs_difference(
        args.full_path, args.none_path, sample_indices, args.time_index
    )
    ratio_field, sample_spatial_mean_ratios = read_mean_relative_difference(
        args.full_path, args.none_path, sample_indices, args.time_index, args.relative_eps
    )
    x_flat, y_flat, z_flat = mesh_coordinates(x, y, z)

    print("=" * 80)
    print("Phi43 Plotly 3D visualization")
    print("=" * 80)
    print("Full path:", args.full_path)
    print("No-renorm path:", args.none_path)
    print("Sample indices:", sample_indices)
    print("Averaging samples:", len(sample_indices) > 1)
    print("Time index:", args.time_index)
    print("Mean field shape:", full_field.shape)
    print("Full attrs renorm_mode:", full_attrs.get("renorm_mode", "unknown"))
    print("No-renorm attrs renorm_mode:", none_attrs.get("renorm_mode", "unknown"))
    print("Full Cmass:", full_cmass)
    print("No-renorm Cmass:", none_cmass)
    print("E[|Diff|] field min/mean/max:", float(diff_field.min()), float(diff_field.mean()), float(diff_field.max()))
    print(
        f"E[|With-No|/(|With|+eps)] field, eps={args.relative_eps}, min/mean/max:",
        float(ratio_field.min()),
        float(ratio_field.mean()),
        float(ratio_field.max()),
    )
    print("Per-sample spatial mean |Diff|:", sample_spatial_mean_diffs.tolist())
    print(
        "Monte Carlo mean +/- std of spatial mean |Diff|:",
        float(sample_spatial_mean_diffs.mean()),
        "+/-",
        float(sample_spatial_mean_diffs.std()),
    )
    print("Per-sample spatial mean relative diff:", sample_spatial_mean_ratios.tolist())
    print(
        "Monte Carlo mean +/- std of spatial mean relative diff:",
        float(sample_spatial_mean_ratios.mean()),
        "+/-",
        float(sample_spatial_mean_ratios.std()),
    )
    print("Output dir:", args.output_dir)

    if len(sample_indices) == 1:
        sample_tag = f"sample{sample_indices[0]}"
        title_sample = f"sample {sample_indices[0]}"
    else:
        sample_tag = f"mean_n{len(sample_indices)}_samples{sample_indices[0]}-{sample_indices[-1]}"
        title_sample = f"mean over {len(sample_indices)} samples"
    base = f"phi43_{sample_tag}_t{args.time_index:03d}_{args.mode}"
    title_suffix = f"{title_sample}, t-index {args.time_index}"

    figures = {
        f"OPEN_THIS_{base}_mean_relative_comparison": make_comparison_figure(
            none_field,
            full_field,
            ratio_field,
            x_flat,
            y_flat,
            z_flat,
            args,
            f"Phi43 renormalization effect ({title_suffix})",
        ),
    }

    if args.write_separate:
        figures.update(
            {
                f"{base}_mean_no_renorm": make_single_figure(
                    none_field,
                    x_flat,
                    y_flat,
                    z_flat,
                    args,
                    f"Phi43 Mean No Renormalization ({title_suffix})",
                    args.none_colorscale,
                ),
                f"{base}_mean_full_renorm": make_single_figure(
                    full_field,
                    x_flat,
                    y_flat,
                    z_flat,
                    args,
                    f"Phi43 Mean Full Renormalization ({title_suffix})",
                    args.full_colorscale,
                ),
                f"{base}_mean_abs_diff": make_single_figure(
                    diff_field,
                    x_flat,
                    y_flat,
                    z_flat,
                    args,
                    f"Phi43 Mean E[|Full - No Renorm|] ({title_suffix})",
                    args.diff_colorscale,
                    positive_only=True,
                ),
                f"{base}_mean_relative_diff": make_single_figure(
                    ratio_field,
                    x_flat,
                    y_flat,
                    z_flat,
                    args,
                    f"Phi43 Mean Relative Difference ({title_suffix})",
                    args.diff_colorscale,
                    positive_only=True,
                ),
            }
        )

    for name, fig in figures.items():
        write_figure(fig, args.output_dir / name)

    print("Done.")


if __name__ == "__main__":
    main()
