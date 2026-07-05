import os
import re
import hashlib

import numpy as np
import scipy.io
import torch

from model.NSPDE.SPDEs2D import SPDE2D


PHI42_SOL_KEYS = ("sol", "sol_reno", "Sol", "Soln_reno")
PHI42_W_KEYS = ("W", "w")
PHI42_AEPS_KEYS = ("a_eps", "aeps", "A_eps")
CACHE_VERSION = 1


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required to read .h5/.hdf5 datasets.") from exc
    return h5py


def resolve_dataset_path(path):
    root, ext = os.path.splitext(str(path))
    if ext:
        return str(path)
    for candidate_ext in (".mat", ".h5", ".hdf5"):
        candidate = root + candidate_ext
        if os.path.exists(candidate):
            return candidate
    return root + ".mat"


def infer_phi42_eps(name):
    match = re.search(r"eps(\d+)", str(name))
    if match is None:
        raise ValueError(f"Could not infer eps from {name!r}; pass Kx_max/Ky_max explicitly.")
    return int(match.group(1))


def _first_key(mapping, candidates):
    for key in candidates:
        if key in mapping:
            return key
    available = ", ".join(str(key) for key in mapping.keys())
    raise KeyError(f"None of {candidates} found. Available keys: {available}")


def _read_mat(path):
    return scipy.io.loadmat(path)


def _cache_key(parts):
    text = "|".join(str(part) for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def default_cache_path(cache_dir, label, source_paths, **params):
    os.makedirs(cache_dir, exist_ok=True)
    key_parts = [label, *source_paths]
    for name in sorted(params):
        key_parts.append(f"{name}={params[name]}")
    return os.path.join(cache_dir, f"{label}_{_cache_key(key_parts)}.h5")


def _sample_to_xyt(sample, t=None):
    sample = np.asarray(sample)
    if sample.ndim != 3:
        raise ValueError(f"Expected one 2D time sample with 3 dims, got {sample.shape}")

    t_len = None if t is None else int(np.asarray(t).squeeze().shape[0])
    if t_len is not None:
        if sample.shape[-1] == t_len:
            return sample
        if sample.shape[0] == t_len:
            return np.transpose(sample, (1, 2, 0))

    if sample.shape[0] <= sample.shape[-1]:
        return np.transpose(sample, (1, 2, 0))
    return sample


def _xyt_to_ntxy_batch(sample_xyt):
    return np.transpose(sample_xyt[None, ...], (0, 3, 1, 2))


class _Raw2DSource:
    def __init__(self, path, equation="Phi42+", include_a_eps=False):
        self.path = str(path)
        self.equation = equation
        self.include_a_eps = include_a_eps
        self.ext = os.path.splitext(self.path)[1].lower()
        self.handle = None
        self.data = None
        self.t = None

        if self.ext in {".h5", ".hdf5"}:
            h5py = _require_h5py()
            self.handle = h5py.File(self.path, "r")
            mapping = self.handle
            if "T" in mapping:
                self.t = np.asarray(mapping["T"][()]).squeeze()
        else:
            self.data = _read_mat(self.path)
            mapping = self.data
            if "T" in mapping:
                self.t = np.asarray(mapping["T"]).squeeze()

        if self.equation == "Phi42+":
            self.w_key = _first_key(mapping, PHI42_W_KEYS)
            self.sol_key = _first_key(mapping, PHI42_SOL_KEYS)
            self.aeps_key = None
            if self.include_a_eps:
                for key in PHI42_AEPS_KEYS:
                    if key in mapping:
                        self.aeps_key = key
                        break
        else:
            self.w_key = _first_key(mapping, ("forcing", "W"))
            self.sol_key = _first_key(mapping, ("sol",))
            self.aeps_key = None

        self.num_samples = int(mapping[self.w_key].shape[0])

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def _array(self, key, index):
        mapping = self.handle if self.handle is not None else self.data
        return np.asarray(mapping[key][index])

    def w_xyt(self, index):
        return _sample_to_xyt(self._array(self.w_key, index), t=self.t)

    def sol_xyt(self, index):
        return _sample_to_xyt(self._array(self.sol_key, index), t=self.t)

    def aeps_xyt(self, index, kx_max=None, ky_max=None):
        if self.aeps_key is not None:
            return _sample_to_xyt(self._array(self.aeps_key, index), t=self.t)
        if kx_max is None:
            kx_max = infer_phi42_eps(self.path)
        w_xyt = self.w_xyt(index)
        a_eps = compute_phi42_aeps(
            _xyt_to_ntxy_batch(w_xyt),
            self.t,
            kx_max=kx_max,
            ky_max=ky_max,
        )[0]
        return _sample_to_xyt(a_eps, t=self.t)


class CachedH5Dataset(torch.utils.data.Dataset):
    def __init__(self, cache_path, split, keys):
        self.cache_path = str(cache_path)
        self.split = split
        self.keys = tuple(keys)
        self._handle = None
        h5py = _require_h5py()
        with h5py.File(self.cache_path, "r") as handle:
            self.length = int(handle[split][self.keys[0]].shape[0])

    def __len__(self):
        return self.length

    def _ensure_handle(self):
        if self._handle is None:
            h5py = _require_h5py()
            self._handle = h5py.File(self.cache_path, "r")
        return self._handle

    def __getitem__(self, index):
        group = self._ensure_handle()[self.split]
        return tuple(torch.from_numpy(group[key][index].astype(np.float32)) for key in self.keys)

    def __del__(self):
        try:
            if self._handle is not None:
                self._handle.close()
                self._handle = None
        except Exception:
            pass


def cached_loader(cache_path, split, keys, batch_size, shuffle=False):
    dataset = CachedH5Dataset(cache_path, split=split, keys=keys)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def _cache_ready(cache_path, expected_splits):
    if not os.path.exists(cache_path):
        return False
    try:
        h5py = _require_h5py()
        with h5py.File(cache_path, "r") as handle:
            if int(handle.attrs.get("cache_version", -1)) != CACHE_VERSION:
                return False
            return all(split in handle for split in expected_splits)
    except OSError:
        return False


def _create_group_datasets(handle, split, count, x_dim, y_dim, t_dim, include_a_eps):
    group = handle.create_group(split)
    compression = "gzip"
    compression_opts = 4
    group.create_dataset(
        "u0",
        shape=(count, 1, x_dim, y_dim),
        dtype=np.float32,
        chunks=(1, 1, x_dim, y_dim),
        compression=compression,
        compression_opts=compression_opts,
    )
    group.create_dataset(
        "xi",
        shape=(count, 1, x_dim, y_dim, t_dim),
        dtype=np.float32,
        chunks=(1, 1, x_dim, y_dim, t_dim),
        compression=compression,
        compression_opts=compression_opts,
    )
    group.create_dataset(
        "u",
        shape=(count, x_dim, y_dim, t_dim),
        dtype=np.float32,
        chunks=(1, x_dim, y_dim, t_dim),
        compression=compression,
        compression_opts=compression_opts,
    )
    if include_a_eps:
        group.create_dataset(
            "a_eps",
            shape=(count, 1, x_dim, y_dim, t_dim),
            dtype=np.float32,
            chunks=(1, 1, x_dim, y_dim, t_dim),
            compression=compression,
            compression_opts=compression_opts,
        )
    return group


def _subsample_xyt(sample_xyt, T, sub_t, sub_x):
    return sample_xyt[::sub_x, ::sub_x, 0:T:sub_t].astype(np.float32, copy=False)


def _write_refs_to_group(group, refs, T, sub_t, sub_x, include_a_eps):
    sources = {}
    aeps_templates = {}
    try:
        for output_index, ref in enumerate(refs):
            path = ref["path"]
            if path not in sources:
                sources[path] = _Raw2DSource(
                    path,
                    equation=ref.get("equation", "Phi42+"),
                    include_a_eps=include_a_eps,
                )
            source = sources[path]
            raw_index = int(ref["index"])
            u = _subsample_xyt(source.sol_xyt(raw_index), T=T, sub_t=sub_t, sub_x=sub_x)
            xi = _subsample_xyt(source.w_xyt(raw_index), T=T, sub_t=sub_t, sub_x=sub_x)
            group["u0"][output_index] = u[:, :, 0][None, ...]
            group["xi"][output_index] = xi[None, ...]
            group["u"][output_index] = u
            if include_a_eps:
                if source.aeps_key is None:
                    eps = ref.get("eps")
                    if eps is None:
                        eps = infer_phi42_eps(path)
                    template_key = (path, int(eps), tuple(xi.shape))
                    if template_key not in aeps_templates:
                        aeps_templates[template_key] = source.aeps_xyt(
                            raw_index,
                            kx_max=eps,
                            ky_max=eps,
                        )
                    a_eps_xyt = aeps_templates[template_key]
                else:
                    a_eps_xyt = source.aeps_xyt(raw_index)
                a_eps = _subsample_xyt(a_eps_xyt, T=T, sub_t=sub_t, sub_x=sub_x)
                group["a_eps"][output_index] = a_eps[None, ...]
    finally:
        for source in sources.values():
            source.close()


def _sample_dims_from_ref(ref, T, sub_t, sub_x, include_a_eps):
    source = _Raw2DSource(
        ref["path"],
        equation=ref.get("equation", "Phi42+"),
        include_a_eps=include_a_eps,
    )
    try:
        u = _subsample_xyt(source.sol_xyt(int(ref["index"])), T=T, sub_t=sub_t, sub_x=sub_x)
        return int(u.shape[0]), int(u.shape[1]), int(u.shape[2])
    finally:
        source.close()


def create_cache_from_refs(cache_path, split_refs, T, sub_t, sub_x, include_a_eps):
    expected_splits = tuple(split_refs.keys())
    if _cache_ready(cache_path, expected_splits):
        print(f"Using existing cache: {cache_path}")
        return cache_path

    all_refs = [ref for refs in split_refs.values() for ref in refs]
    if not all_refs:
        raise ValueError("Cannot create cache without any samples.")

    h5py = _require_h5py()
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    x_dim, y_dim, t_dim = _sample_dims_from_ref(
        all_refs[0],
        T=T,
        sub_t=sub_t,
        sub_x=sub_x,
        include_a_eps=include_a_eps,
    )
    print(f"Creating cache: {cache_path}")
    with h5py.File(cache_path, "w") as handle:
        handle.attrs["cache_version"] = CACHE_VERSION
        handle.attrs["T"] = int(T)
        handle.attrs["sub_t"] = int(sub_t)
        handle.attrs["sub_x"] = int(sub_x)
        handle.attrs["include_a_eps"] = bool(include_a_eps)
        for split, refs in split_refs.items():
            group = _create_group_datasets(
                handle,
                split=split,
                count=len(refs),
                x_dim=x_dim,
                y_dim=y_dim,
                t_dim=t_dim,
                include_a_eps=include_a_eps,
            )
            _write_refs_to_group(
                group,
                refs=refs,
                T=T,
                sub_t=sub_t,
                sub_x=sub_x,
                include_a_eps=include_a_eps,
            )
    return cache_path


def build_sequential_refs(path, count=None, equation="Phi42+", eps=None, start=0):
    source = _Raw2DSource(path, equation=equation, include_a_eps=False)
    try:
        available = source.num_samples - int(start)
    finally:
        source.close()
    if count is None:
        count = available
    count = min(int(count), int(available))
    return [
        {"path": str(path), "index": int(start + index), "equation": equation, "eps": eps}
        for index in range(count)
    ]


def shuffled_alleps_refs(paths_by_eps, samples_per_eps=240, sample_size=1200, seed=3407):
    refs = []
    for eps, path in paths_by_eps:
        refs.extend(
            build_sequential_refs(
                path,
                count=samples_per_eps,
                equation="Phi42+",
                eps=int(eps),
            )
        )
    sample_size = min(int(sample_size), len(refs))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    perm = torch.randperm(len(refs), generator=generator)[:sample_size].tolist()
    return [refs[index] for index in perm]


def split_train_val_test_refs(refs, ntrain, nval, ntest):
    return {
        "train": refs[: int(ntrain)],
        "val": refs[int(ntrain) : int(ntrain) + int(nval)],
        "test": refs[-int(ntest) :],
    }


def validate_full_batches(split_refs, batch_size):
    batch_size = int(batch_size)
    for split, refs in split_refs.items():
        if len(refs) == 0:
            raise ValueError(f"Cache split {split!r} has no samples.")
        if len(refs) % batch_size != 0:
            raise ValueError(
                f"Cache split {split!r} has {len(refs)} samples, which is not divisible "
                f"by batch_size={batch_size}. The existing loss code reshapes with a fixed "
                "batch size, so adjust the sample count or batch_size."
            )


def _read_h5_array(handle, candidates, sample_count=None):
    key = _first_key(handle, candidates)
    dataset = handle[key]
    if sample_count is None or dataset.ndim == 0:
        return dataset[()]
    return dataset[:sample_count, ...]


def _read_mat_array(data, candidates, sample_count=None):
    key = _first_key(data, candidates)
    array = data[key]
    if sample_count is not None and array.ndim > 0:
        array = array[:sample_count, ...]
    return array


def read_phi42_raw(path, sample_count=None, include_a_eps=True):
    ext = os.path.splitext(path)[1].lower()
    if ext in {".h5", ".hdf5"}:
        h5py = _require_h5py()
        with h5py.File(path, "r") as handle:
            x = np.asarray(handle["X"][()]).squeeze()
            y = np.asarray(handle["Y"][()]).squeeze()
            t = np.asarray(handle["T"][()]).squeeze()
            w = np.asarray(_read_h5_array(handle, PHI42_W_KEYS, sample_count=sample_count))
            sol = np.asarray(_read_h5_array(handle, PHI42_SOL_KEYS, sample_count=sample_count))
            a_eps = None
            if include_a_eps and any(key in handle for key in PHI42_AEPS_KEYS):
                a_eps = np.asarray(
                    _read_h5_array(handle, PHI42_AEPS_KEYS, sample_count=sample_count)
                )
    else:
        data = _read_mat(path)
        x = np.asarray(data["X"]).squeeze()
        y = np.asarray(data["Y"]).squeeze()
        t = np.asarray(data["T"]).squeeze()
        w = np.asarray(_read_mat_array(data, PHI42_W_KEYS, sample_count=sample_count))
        sol = np.asarray(_read_mat_array(data, PHI42_SOL_KEYS, sample_count=sample_count))
        a_eps = None
        if include_a_eps and any(key in data for key in PHI42_AEPS_KEYS):
            a_eps = np.asarray(
                _read_mat_array(data, PHI42_AEPS_KEYS, sample_count=sample_count)
            )
    return x, y, t, w, sol, a_eps


def read_2d_raw(path, equation, sample_count=None):
    if equation == "Phi42+":
        _, _, _, w, sol, _ = read_phi42_raw(
            path, sample_count=sample_count, include_a_eps=False
        )
        return w, sol

    ext = os.path.splitext(path)[1].lower()
    if ext in {".h5", ".hdf5"}:
        h5py = _require_h5py()
        with h5py.File(path, "r") as handle:
            w = np.asarray(_read_h5_array(handle, ("forcing", "W"), sample_count=sample_count))
            sol = np.asarray(_read_h5_array(handle, ("sol",), sample_count=sample_count))
    else:
        data = _read_mat(path)
        w = np.asarray(_read_mat_array(data, ("forcing", "W"), sample_count=sample_count))
        sol = np.asarray(_read_mat_array(data, ("sol",), sample_count=sample_count))
    return w, sol


def to_nxyt(array, t=None):
    array = np.asarray(array)
    if array.ndim != 4:
        return array

    t_len = None if t is None else int(np.asarray(t).squeeze().shape[0])
    if t_len is not None:
        if array.shape[-1] == t_len:
            return array
        if array.shape[1] == t_len:
            return np.transpose(array, (0, 2, 3, 1))

    if array.shape[1] >= array.shape[-1]:
        return np.transpose(array, (0, 2, 3, 1))
    return array


def to_ntxy(array, t=None):
    array = np.asarray(array)
    if array.ndim != 4:
        return array

    t_len = None if t is None else int(np.asarray(t).squeeze().shape[0])
    if t_len is not None:
        if array.shape[1] == t_len:
            return array
        if array.shape[-1] == t_len:
            return np.transpose(array, (0, 3, 1, 2))

    if array.shape[-1] >= array.shape[1]:
        return np.transpose(array, (0, 3, 1, 2))
    return array


def compute_phi42_aeps(w_ntxy, t, kx_max, ky_max=None):
    ky_max = kx_max if ky_max is None else ky_max
    a_eps = SPDE2D().Renormalize_Constant(
        np.asarray(w_ntxy),
        np.asarray(t).squeeze(),
        Kx_max=int(kx_max),
        Ky_max=int(ky_max),
    )
    return np.repeat(a_eps[None, ...], int(w_ntxy.shape[0]), axis=0)


def load_nspde_2d_arrays(path, equation, sample_count=None):
    w, sol = read_2d_raw(path, equation=equation, sample_count=sample_count)
    if equation == "Phi42+":
        _, _, t, _, _, _ = read_phi42_raw(path, sample_count=0, include_a_eps=False)
        w = to_nxyt(w, t=t)
        sol = to_nxyt(sol, t=t)
    return w.astype(np.float32), sol.astype(np.float32)


def load_nspde_2d_aeps_arrays(path, kx_max=None, ky_max=None, sample_count=None):
    _, _, t, w, sol, a_eps = read_phi42_raw(
        path, sample_count=sample_count, include_a_eps=True
    )
    w_ntxy = to_ntxy(w, t=t)
    sol_ntxy = to_ntxy(sol, t=t)
    if a_eps is None:
        if kx_max is None:
            kx_max = infer_phi42_eps(path)
        a_eps = compute_phi42_aeps(w_ntxy, t, kx_max=kx_max, ky_max=ky_max)
    else:
        a_eps = to_ntxy(a_eps, t=t)

    return (
        torch.from_numpy(to_nxyt(w_ntxy, t=t).astype(np.float32)),
        torch.from_numpy(to_nxyt(sol_ntxy, t=t).astype(np.float32)),
        torch.from_numpy(to_nxyt(a_eps, t=t).astype(np.float32)),
    )


def load_alleps_from_paths(paths_by_eps, samples_per_eps=240, sample_size=1200, seed=3407):
    xs, ys, us, aepss = [], [], [], []
    for eps, path in paths_by_eps:
        xi, u, a_eps = load_nspde_2d_aeps_arrays(
            path, kx_max=int(eps), ky_max=int(eps), sample_count=samples_per_eps
        )
        xs.append(xi)
        us.append(u)
        aepss.append(a_eps)

    xi = torch.cat(xs, dim=0)
    u = torch.cat(us, dim=0)
    a_eps = torch.cat(aepss, dim=0)
    sample_size = min(int(sample_size), int(u.shape[0]))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    perm = torch.randperm(int(u.shape[0]), generator=generator)[:sample_size]
    return xi[perm, ...], u[perm, ...], a_eps[perm, ...]
