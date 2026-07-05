"""Merge five Phi42+ eps H5/MAT files into one mixed H5 dataset.

Usage: python data_gen/examples/merge_phi42_alleps.py --inputs eps2.h5 eps8.h5 eps32.h5 eps64.h5 eps128.h5 --output mixed.h5
"""

import argparse
import os
import os.path as osp
import sys

import numpy as np
import scipy.io

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(osp.join(current_directory, "..", ".."))

from model.NSPDE.data_io import compute_phi42_aeps, read_phi42_raw, to_ntxy


DEFAULT_EPS = (2, 8, 32, 64, 128)


def _samples_per_input(sample_size, n_inputs):
    base = sample_size // n_inputs
    remainder = sample_size % n_inputs
    return [base + int(index < remainder) for index in range(n_inputs)]


def _check_grid(name, reference, current):
    if reference.shape != current.shape or not np.allclose(reference, current):
        raise ValueError(f"{name} grids differ across input files.")


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required for streaming .h5 output.") from exc
    return h5py


def _first_key(mapping, candidates):
    for key in candidates:
        if key in mapping:
            return key
    available = ", ".join(str(key) for key in mapping.keys())
    raise KeyError(f"None of {candidates} found. Available keys: {available}")


def _read_source_metadata(path):
    ext = osp.splitext(path)[1].lower()
    if ext in {".h5", ".hdf5"}:
        h5py = _require_h5py()
        with h5py.File(path, "r") as handle:
            w_key = _first_key(handle, ("W", "w"))
            sol_key = _first_key(handle, ("sol", "sol_reno", "Sol", "Soln_reno"))
            x = np.asarray(handle["X"][()]).squeeze()
            y = np.asarray(handle["Y"][()]).squeeze()
            t = np.asarray(handle["T"][()]).squeeze()
            w_shape = tuple(handle[w_key].shape)
            sol_shape = tuple(handle[sol_key].shape)
        return x, y, t, w_key, sol_key, w_shape, sol_shape

    data = scipy.io.loadmat(path)
    w_key = _first_key(data, ("W", "w"))
    sol_key = _first_key(data, ("sol", "sol_reno", "Sol", "Soln_reno"))
    x = np.asarray(data["X"]).squeeze()
    y = np.asarray(data["Y"]).squeeze()
    t = np.asarray(data["T"]).squeeze()
    return x, y, t, w_key, sol_key, tuple(data[w_key].shape), tuple(data[sol_key].shape)


def _slice_source(path, key, start, end):
    ext = osp.splitext(path)[1].lower()
    if ext in {".h5", ".hdf5"}:
        h5py = _require_h5py()
        with h5py.File(path, "r") as handle:
            return np.asarray(handle[key][start:end, ...])
    data = scipy.io.loadmat(path)
    return np.asarray(data[key][start:end, ...])


def _save_h5(path, arrays, compression="gzip", compression_opts=4):
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required to write .h5/.hdf5 datasets.") from exc

    os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)
    with h5py.File(path, "w") as handle:
        for key in ("X", "Y", "T", "eps_values", "samples_per_eps"):
            handle.create_dataset(key, data=arrays[key])

        for key in ("W", "sol", "a_eps"):
            value = np.asarray(arrays[key])
            kwargs = {"chunks": (1,) + value.shape[1:]}
            if compression is not None:
                kwargs["compression"] = compression
                kwargs["compression_opts"] = compression_opts
            handle.create_dataset(
                key,
                data=value,
                **kwargs,
            )

        handle["sol_reno"] = handle["sol"]
        handle.attrs["generated_by"] = "merge_phi42_alleps.py"
        handle.attrs["total_size"] = int(arrays["W"].shape[0])


def _save_mat(path, arrays):
    os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)
    scipy.io.savemat(
        path,
        {
            "X": arrays["X"],
            "Y": arrays["Y"],
            "T": arrays["T"],
            "W": arrays["W"],
            "sol": arrays["sol"],
            "a_eps": arrays["a_eps"],
            "eps_values": arrays["eps_values"],
            "samples_per_eps": arrays["samples_per_eps"],
        },
        do_compression=True,
    )


def _aeps_template(w_one_ntxy, t, eps):
    return compute_phi42_aeps(w_one_ntxy[:1], t, kx_max=int(eps), ky_max=int(eps))[0].astype(
        np.float32
    )


def _create_output_h5(path, x, y, t, sample_size, sample_shape, eps_values, per_input, compression):
    h5py = _require_h5py()
    os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)
    handle = h5py.File(path, "w")
    handle.create_dataset("X", data=x)
    handle.create_dataset("Y", data=y)
    handle.create_dataset("T", data=t)
    handle.create_dataset("eps_values", data=np.asarray(eps_values, dtype=np.int32))
    handle.create_dataset("samples_per_eps", data=np.asarray(per_input, dtype=np.int32))
    handle.create_dataset("source_eps", shape=(sample_size,), dtype=np.int32)

    kwargs = {"chunks": (1,) + sample_shape}
    if compression is not None:
        kwargs["compression"] = compression
        kwargs["compression_opts"] = 4
    shape = (sample_size,) + sample_shape
    handle.create_dataset("W", shape=shape, dtype=np.float32, **kwargs)
    handle.create_dataset("sol", shape=shape, dtype=np.float32, **kwargs)
    handle.create_dataset("a_eps", shape=shape, dtype=np.float32, **kwargs)
    handle["sol_reno"] = handle["sol"]
    handle.attrs["generated_by"] = "merge_phi42_alleps.py"
    handle.attrs["total_size"] = int(sample_size)
    return handle


def _merge_phi42_alleps_h5(
    input_paths,
    output_path,
    eps_values,
    sample_size,
    seed,
    compression,
    chunk_size,
):
    per_input = _samples_per_input(sample_size, len(input_paths))
    x_ref = y_ref = t_ref = None
    metadata = []

    for path, eps, take_count in zip(input_paths, eps_values, per_input):
        x, y, t, w_key, sol_key, w_shape, sol_shape = _read_source_metadata(path)
        if w_shape[0] < take_count:
            raise ValueError(f"{path} only has {w_shape[0]} samples, need {take_count}.")
        if sol_shape[0] < take_count:
            raise ValueError(f"{path} only has {sol_shape[0]} solution samples, need {take_count}.")
        if x_ref is None:
            x_ref, y_ref, t_ref = x, y, t
        else:
            _check_grid("X", x_ref, x)
            _check_grid("Y", y_ref, y)
            _check_grid("T", t_ref, t)
        metadata.append((path, int(eps), int(take_count), w_key, sol_key))

    first_path, first_eps, _, first_w_key, _ = metadata[0]
    first_w = to_ntxy(_slice_source(first_path, first_w_key, 0, 1), t=t_ref).astype(np.float32)
    sample_shape = tuple(first_w.shape[1:])

    rng = np.random.default_rng(seed)
    perm = rng.permutation(sample_size)
    out = _create_output_h5(
        output_path,
        x_ref,
        y_ref,
        t_ref,
        sample_size,
        sample_shape,
        eps_values,
        per_input,
        compression,
    )

    try:
        source_offset = 0
        for path, eps, take_count, w_key, sol_key in metadata:
            print(f"Streaming {take_count} samples from eps={eps}: {path}")
            template = None
            for start in range(0, take_count, chunk_size):
                end = min(start + chunk_size, take_count)
                w = to_ntxy(_slice_source(path, w_key, start, end), t=t_ref).astype(
                    np.float32, copy=False
                )
                sol = to_ntxy(_slice_source(path, sol_key, start, end), t=t_ref).astype(
                    np.float32, copy=False
                )
                if tuple(w.shape[1:]) != sample_shape:
                    raise ValueError(
                        f"{path} has sample shape {tuple(w.shape[1:])}, expected {sample_shape}."
                    )
                if template is None:
                    template = _aeps_template(w[:1], t_ref, eps)

                target = perm[source_offset + start : source_offset + end]
                order = np.argsort(target)
                target_sorted = target[order]
                out["W"][target_sorted, ...] = w[order]
                out["sol"][target_sorted, ...] = sol[order]
                out["a_eps"][target_sorted, ...] = np.broadcast_to(
                    template, (len(target_sorted),) + sample_shape
                )
                out["source_eps"][target_sorted] = eps
                print(f"  wrote samples {start}:{end}")
            source_offset += take_count
    finally:
        out.close()

    print(f"Saved merged dataset to {output_path}")
    print("W shape:", (sample_size,) + sample_shape)
    print("sol shape:", (sample_size,) + sample_shape)
    print("a_eps shape:", (sample_size,) + sample_shape)


def merge_phi42_alleps(input_paths, output_path, eps_values, sample_size, seed, compression, chunk_size):
    if len(input_paths) != len(eps_values):
        raise ValueError("input_paths and eps_values must have the same length.")

    ext = osp.splitext(output_path)[1].lower()
    if ext in {".h5", ".hdf5"}:
        _merge_phi42_alleps_h5(
            input_paths=input_paths,
            output_path=output_path,
            eps_values=eps_values,
            sample_size=sample_size,
            seed=seed,
            compression=compression,
            chunk_size=chunk_size,
        )
        return

    per_input = _samples_per_input(sample_size, len(input_paths))
    x_ref = y_ref = t_ref = None
    w_blocks = []
    sol_blocks = []
    aeps_blocks = []

    for path, eps, take_count in zip(input_paths, eps_values, per_input):
        print(f"Reading {take_count} samples from eps={eps}: {path}")
        x, y, t, w, sol, _ = read_phi42_raw(
            path, sample_count=take_count, include_a_eps=False
        )
        w = to_ntxy(w, t=t).astype(np.float32)
        sol = to_ntxy(sol, t=t).astype(np.float32)

        if x_ref is None:
            x_ref, y_ref, t_ref = x, y, t
        else:
            _check_grid("X", x_ref, x)
            _check_grid("Y", y_ref, y)
            _check_grid("T", t_ref, t)

        if w.shape[0] < take_count:
            raise ValueError(f"{path} only has {w.shape[0]} samples, need {take_count}.")

        a_eps = compute_phi42_aeps(w, t, kx_max=int(eps), ky_max=int(eps)).astype(np.float32)
        w_blocks.append(w)
        sol_blocks.append(sol)
        aeps_blocks.append(a_eps)

    w_all = np.concatenate(w_blocks, axis=0)
    sol_all = np.concatenate(sol_blocks, axis=0)
    aeps_all = np.concatenate(aeps_blocks, axis=0)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(w_all.shape[0])[:sample_size]
    arrays = {
        "X": x_ref,
        "Y": y_ref,
        "T": t_ref,
        "W": w_all[perm],
        "sol": sol_all[perm],
        "a_eps": aeps_all[perm],
        "eps_values": np.asarray(eps_values, dtype=np.int32),
        "samples_per_eps": np.asarray(per_input, dtype=np.int32),
    }

    if ext == ".mat":
        _save_mat(output_path, arrays)
    else:
        raise ValueError("output_path must end with .h5, .hdf5, or .mat")

    print(f"Saved merged dataset to {output_path}")
    print("W shape:", arrays["W"].shape)
    print("sol shape:", arrays["sol"].shape)
    print("a_eps shape:", arrays["a_eps"].shape)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge Phi42 eps=2/8/32/64/128 files before NSPDE training."
    )
    parser.add_argument(
        "--inputs",
        nargs=5,
        required=True,
        help="Five local .mat/.h5 files, normally ordered as eps 2 8 32 64 128.",
    )
    parser.add_argument("--output", required=True, help="Merged output path (.h5 or .mat).")
    parser.add_argument("--eps", nargs=5, type=int, default=DEFAULT_EPS)
    parser.add_argument("--sample-size", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--compression", default="gzip", help="HDF5 compression, or 'none'.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4,
        help="Number of samples to stream at a time when writing .h5 output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    compression = None if str(args.compression).lower() == "none" else args.compression
    merge_phi42_alleps(
        input_paths=args.inputs,
        output_path=args.output,
        eps_values=args.eps,
        sample_size=args.sample_size,
        seed=args.seed,
        compression=compression,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
