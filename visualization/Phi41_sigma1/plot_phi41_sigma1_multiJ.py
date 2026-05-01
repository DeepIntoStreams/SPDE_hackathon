import argparse
import os
import os.path as osp
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io


CURRENT_DIR = osp.dirname(osp.abspath(__file__))


def parse_j_from_name(path):
    match = re.search(r"_trc(\d+)_", osp.basename(path))
    if match is None:
        raise ValueError(f"Could not parse truncation J from filename: {path}")
    return int(match.group(1))


def load_dataset(path):
    data = scipy.io.loadmat(path)
    t = np.asarray(data["T"]).reshape(-1)
    w = np.asarray(data["W"], dtype=np.float64)
    sol = np.asarray(data["sol"], dtype=np.float64)
    if w.ndim != 3 or sol.ndim != 3:
        raise ValueError(f"Expected W/sol shape [N, X, T], got W={w.shape}, sol={sol.shape}")
    return t, w, sol


def load_all(data_dir, pattern, j_values):
    files = []
    for name in os.listdir(data_dir):
        if name.endswith(".mat") and pattern in name:
            files.append(osp.join(data_dir, name))
    if not files:
        raise FileNotFoundError(f"No .mat files containing '{pattern}' found in {data_dir}")

    by_j = {parse_j_from_name(path): path for path in files}
    selected = {}
    for j in j_values:
        if j not in by_j:
            raise FileNotFoundError(f"Missing file for J={j} in {data_dir}")
        selected[j] = by_j[j]
    return selected


def compute_stats(t, w, sol):
    mean_t = sol.mean(axis=(0, 1))
    std_t = sol.std(axis=(0, 1))

    energy_paths = (sol**2).mean(axis=1)
    energy_mean = energy_paths.mean(axis=0)
    energy_std = energy_paths.std(axis=0)

    final_values = sol[:, :, -1].reshape(-1)
    noise_std_t = w.std(axis=(0, 1))

    global_std = sol.std()
    final_energy = energy_paths[:, -1].mean()

    return {
        "t": t,
        "mean_t": mean_t,
        "std_t": std_t,
        "energy_mean": energy_mean,
        "energy_std": energy_std,
        "final_values": final_values,
        "noise_std_t": noise_std_t,
        "global_std": global_std,
        "final_energy": final_energy,
    }


def setup_style():
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 19,
            "axes.titleweight": "bold",
            "axes.labelsize": 18,
            "axes.labelweight": "bold",
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "lines.linewidth": 2.8,
            "axes.linewidth": 1.4,
        }
    )


def plot_multi_panel(stats_by_j, out_path, title=None, dpi=300):
    setup_style()
    js = list(stats_by_j.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(js)))
    labels = {j: rf"$J={j}$" for j in js}

    fig, axes = plt.subplots(2, 3, figsize=(20, 11), constrained_layout=True)
    ax = axes.ravel()

    for color, j in zip(colors, js):
        stats = stats_by_j[j]
        t = stats["t"]
        ax[0].plot(t, stats["mean_t"], color=color, label=labels[j])
        ax[1].plot(t, stats["std_t"], color=color, label=labels[j])
        ax[2].plot(t, stats["energy_mean"], color=color, label=labels[j])
        ax[2].fill_between(
            t,
            stats["energy_mean"] - stats["energy_std"],
            stats["energy_mean"] + stats["energy_std"],
            color=color,
            alpha=0.14,
            linewidth=0,
        )
        ax[3].hist(
            stats["final_values"],
            bins=80,
            density=True,
            histtype="stepfilled",
            alpha=0.18,
            color=color,
            label=labels[j],
        )
        ax[3].hist(
            stats["final_values"],
            bins=80,
            density=True,
            histtype="step",
            linewidth=1.5,
            color=color,
        )
        ax[4].plot(t, stats["noise_std_t"], color=color, label=labels[j])

    x = np.arange(len(js))
    width = 0.36
    global_stds = [stats_by_j[j]["global_std"] for j in js]
    final_energies = [stats_by_j[j]["final_energy"] for j in js]
    ax[5].bar(x - width / 2, global_stds, width, label="global std", color=colors, alpha=0.75)
    ax[5].bar(
        x + width / 2,
        final_energies,
        width,
        label=r"$\mathbf{E[u(T)^2]}$",
        color=colors,
        alpha=0.35,
        hatch="//",
        edgecolor="black",
        linewidth=0.4,
    )
    ax[5].set_xticks(x)
    ax[5].set_xticklabels([rf"$\mathbf{{J={j}}}$" for j in js], rotation=0, ha="center")

    panel_titles = [
        "(a) Spatial-sample mean vs time",
        "(b) Spatial-sample std vs time",
        r"(c) Energy $\mathbf{E[u^2]}$ vs time",
        "(d) Distribution of final-time field values",
        "(e) Noise std vs time",
        "(f) Global std and final energy",
    ]
    ylabels = [
        r"$\mathbf{\langle u\rangle}_{\mathbf{x},\mathbf{samples}}$",
        r"$\mathbf{std(u)}$",
        r"$\mathbf{\langle u^2\rangle}_{\mathbf{x}}$",
        "density",
        r"$\mathbf{std(W)}$",
        "value",
    ]

    for i, axis in enumerate(ax):
        axis.set_title(panel_titles[i], pad=12, fontsize=19, fontweight="bold")
        axis.set_xlabel(r"$t$", fontsize=18, fontweight="bold")
        axis.set_ylabel(ylabels[i], fontsize=18, fontweight="bold")
        axis.grid(True, alpha=0.28, linewidth=0.8)
        for tick in axis.get_xticklabels() + axis.get_yticklabels():
            tick.set_fontsize(14)
            tick.set_fontweight("bold")

    ax[3].set_xlabel(r"$\mathbf{u(x,T)}$", fontsize=18, fontweight="bold")
    ax[5].set_xlabel("")

    for i, axis in enumerate(ax[:5]):
        if i == 3:
            axis.legend(
                ncol=2,
                frameon=True,
                framealpha=0.9,
                prop={"size": 12, "weight": "bold"},
                loc="upper left",
            )
        else:
            axis.legend(ncol=2, frameon=True, framealpha=0.9, prop={"size": 13, "weight": "bold"})
    ax[5].legend(frameon=True, framealpha=0.9, prop={"size": 13, "weight": "bold"})

    if title:
        fig.suptitle(title, fontsize=18, fontweight="bold")

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Phi41 sigma=1 statistics across truncation J.")
    parser.add_argument("--data-dir", type=str, default=osp.join(CURRENT_DIR, "data"))
    parser.add_argument("--pattern", type=str, default="Phi41_sigma1_xi")
    parser.add_argument("--j-values", type=int, nargs="+", default=[2, 8, 32, 64, 128, 256])
    parser.add_argument("--out", type=str, default=osp.join(CURRENT_DIR, "phi41_sigma1_multiJ_stats.png"))
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    paths_by_j = load_all(args.data_dir, args.pattern, args.j_values)

    stats_by_j = {}
    reference_t = None
    for j, path in paths_by_j.items():
        print(f"Loading J={j}: {path}")
        t, w, sol = load_dataset(path)
        if reference_t is None:
            reference_t = t
        elif len(t) != len(reference_t) or np.max(np.abs(t - reference_t)) > 1e-12:
            raise ValueError(f"Time grid mismatch for J={j}")
        stats_by_j[j] = compute_stats(t, w, sol)

    plot_multi_panel(stats_by_j, args.out, title=args.title, dpi=args.dpi)
    if args.pdf:
        pdf_out = osp.splitext(args.out)[0] + ".pdf"
        plot_multi_panel(stats_by_j, pdf_out, title=args.title, dpi=args.dpi)


if __name__ == "__main__":
    main()
