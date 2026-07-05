"""Train or evaluate the 2D NSPDE model without a_eps input.

Usage: python model/NSPDE/train2d.py mode=train_1200 to train on 1200 samples.
Usage: python model/NSPDE/train2d.py mode=eval_mean_std to test a checkpoint on five splits.
"""

import csv
import itertools
import os
import os.path as osp
import random
import statistics
import sys

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(osp.join(current_directory, "..", ".."))

from model.NSPDE.data_io import (
    build_sequential_refs,
    cached_loader,
    create_cache_from_refs,
    default_cache_path,
    resolve_dataset_path,
    split_train_val_test_refs,
    validate_full_batches,
)
from model.NSPDE.utilities import LpLoss, eval_nspde, train_nspde
from model.NSPDE.neural_spde import NeuralSPDE
from model.utilities import count_params


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _mode(config):
    return str(getattr(config, "mode", "eval_mean_std"))


def _equation(config):
    return "Phi42+" if config.equation in {"phi42", "Phi42+"} else "NS+"


def _cache_dir(config):
    return getattr(config, "cache_dir", None) or osp.join(config.save_dir, "cache")


def _run_name(config, suffix):
    data = str(getattr(config, "data", "data"))
    return f"{suffix} {data}"


def _init_wandb(config, suffix):
    if wandb.run is not None:
        return
    project_name = (
        "NSPDE2D_refined"
        + str(config.equation).lower().replace("+", "").replace(" ", "").replace("-", "_")
    )
    wandb.init(
        entity="SPDE_hackathon",
        project=project_name,
        config=OmegaConf.to_container(config, resolve=True),
        name=_run_name(config, suffix),
    )
    wandb.define_metric("epoch")
    wandb.define_metric("train_loss", step_metric="epoch")
    wandb.define_metric("val_loss", step_metric="epoch")


def _wandb_update_config(values):
    if wandb.run is not None:
        wandb.config.update(values, allow_val_change=True)


def _wandb_log_event(values):
    if wandb.run is not None:
        wandb.log(values)


def _log_training_loss_plots(losses_train, losses_val, print_every):
    if wandb.run is None:
        return
    if not losses_train or not losses_val:
        return

    count = min(len(losses_train), len(losses_val))
    epochs = [index * int(print_every) for index in range(count)]
    train_values = [float(value) for value in losses_train[:count]]
    val_values = [float(value) for value in losses_val[:count]]

    table = wandb.Table(columns=["epoch", "train_loss", "val_loss"])
    for epoch, train_loss, val_loss in zip(epochs, train_values, val_values):
        table.add_data(epoch, train_loss, val_loss)

    wandb.log(
        {
            "loss_curve_table": table,
            "train_val_loss_curve": wandb.plot.line_series(
                xs=epochs,
                ys=[train_values, val_values],
                keys=["train_loss", "val_loss"],
                title="Train and Validation Loss",
                xname="epoch",
            ),
        }
    )


def _log_final_loss_plot(loss_train, loss_val, loss_test):
    if wandb.run is None:
        return
    table = wandb.Table(columns=["metric", "loss"])
    rows = [
        ("train_loss", float(loss_train)),
        ("val_loss", float(loss_val)),
        ("test_loss", float(loss_test)),
    ]
    for metric, value in rows:
        table.add_data(metric, value)

    wandb.log(
        {
            "final_train_loss": float(loss_train),
            "final_val_loss": float(loss_val),
            "final_test_loss": float(loss_test),
            "final_loss_table": table,
            "final_loss_bar": wandb.plot.bar(
                table,
                "metric",
                "loss",
                title="Final Train / Val / Test Loss",
            ),
        }
    )


def _log_eval_mean_std_plot(split_losses, mean_loss, std_loss):
    if wandb.run is None:
        return
    table = wandb.Table(columns=["split", "test_loss"])
    for split, value in split_losses:
        table.add_data(split, float(value))

    wandb.log(
        {
            "mean_test_loss": float(mean_loss),
            "std_test_loss": float(std_loss),
            "eval_split_loss_table": table,
            "eval_split_loss_bar": wandb.plot.bar(
                table,
                "split",
                "test_loss",
                title="5 Split Test Loss",
            ),
        }
    )


def _build_model(config):
    model = NeuralSPDE(
        dim=2,
        in_channels=1,
        noise_channels=1,
        hidden_channels=config.hidden_channels,
        n_iter=config.n_iter,
        modes1=config.modes1,
        modes2=config.modes2,
        modes3=config.modes3,
        solver=config.solver,
    ).cuda()
    print("The model has {} parameters".format(count_params(model)))
    return model


def _make_cache_loaders(config, split_refs, label, source_paths, shuffle_train=False):
    validate_full_batches(split_refs, config.batch_size)
    cache_path = default_cache_path(
        _cache_dir(config),
        label,
        source_paths,
        T=config.T,
        sub_t=config.sub_t,
        sub_x=config.sub_x,
        batch_size=config.batch_size,
        splits=",".join(split_refs.keys()),
        counts=",".join(str(len(refs)) for refs in split_refs.values()),
    )
    _wandb_update_config(
        {
            "cache_path": cache_path,
            "cache_label": label,
            "cache_source_paths": list(source_paths),
            "cache_split_sizes": {split: len(refs) for split, refs in split_refs.items()},
        }
    )
    _wandb_log_event({"stage/cache_create_start": 1})
    create_cache_from_refs(
        cache_path,
        split_refs,
        T=config.T,
        sub_t=config.sub_t,
        sub_x=config.sub_x,
        include_a_eps=False,
    )
    _wandb_log_event({"stage/cache_create_done": 1})
    keys = ("u0", "xi", "u")
    loaders = {
        split: cached_loader(
            cache_path,
            split,
            keys,
            config.batch_size,
            shuffle=(shuffle_train and split == "train"),
        )
        for split in split_refs
    }
    return loaders, cache_path


def _training_refs(config, data_path):
    sample_size = int(config.ntrain) + int(config.nval) + int(config.ntest)
    refs = build_sequential_refs(data_path, count=sample_size, equation=_equation(config))
    indices = np.random.permutation(len(refs))
    print("indices:", indices[:10])
    return [refs[int(index)] for index in indices]


def _eval_split_refs(config, data_path):
    samples_per_split = int(getattr(config, "eval_samples_per_split", 500))
    eval_data_path = getattr(config, "eval_data_path", None)
    if eval_data_path:
        data_path = resolve_dataset_path(eval_data_path)
    refs = build_sequential_refs(
        data_path,
        count=5 * samples_per_split,
        equation=_equation(config),
    )
    return {
        f"test{index + 1}": refs[index * samples_per_split : (index + 1) * samples_per_split]
        for index in range(5)
    }


def train_on_1200(config):
    _init_wandb(config, "train")
    os.makedirs(config.save_dir, exist_ok=True)
    checkpoint_file = osp.join(config.save_dir, config.checkpoint_file)
    data_path = resolve_dataset_path(config.data_path)
    _wandb_update_config(
        {
            "resolved_data_path": data_path,
            "checkpoint_file": checkpoint_file,
            "run_mode": "train_1200",
        }
    )
    _wandb_log_event({"stage/run_start": 1})

    split_refs = split_train_val_test_refs(
        _training_refs(config, data_path),
        ntrain=config.ntrain,
        nval=config.nval,
        ntest=config.ntest,
    )
    loaders, cache_path = _make_cache_loaders(
        config,
        split_refs,
        label="nspde_2d_train_1200",
        source_paths=[data_path],
        shuffle_train=True,
    )

    _wandb_log_event({"stage/loaders_ready": 1, "cache_file_size_mb": os.path.getsize(cache_path) / (1024 ** 2)})
    model = _build_model(config)
    wandb.watch(model, log="gradients", log_freq=max(int(config.print_every), 1))
    loss = LpLoss(size_average=False)
    _wandb_log_event({"stage/training_start": 1})

    model, losses_train, losses_val = train_nspde(
        model,
        loaders["train"],
        loaders["val"],
        device,
        loss,
        batch_size=config.batch_size,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        plateau_patience=config.plateau_patience,
        factor=getattr(config, "factor", 0.1),
        delta=config.delta,
        plateau_terminate=config.plateau_terminate,
        print_every=config.print_every,
        checkpoint_file=checkpoint_file,
    )
    _wandb_log_event({"stage/training_done": 1})
    _log_training_loss_plots(losses_train, losses_val, config.print_every)

    model.load_state_dict(torch.load(checkpoint_file))
    _wandb_log_event({"stage/final_eval_start": 1})
    loss_train = eval_nspde(model, loaders["train"], loss, config.batch_size, device)
    loss_val = eval_nspde(model, loaders["val"], loss, config.batch_size, device)
    loss_test = eval_nspde(model, loaders["test"], loss, config.batch_size, device)
    print("loss_train (model saved in checkpoint):", loss_train)
    print("loss_val (model saved in checkpoint):", loss_val)
    print("loss_test (model saved in checkpoint):", loss_test)
    _log_final_loss_plot(loss_train, loss_val, loss_test)
    _wandb_log_event({"stage/run_done": 1})
    wandb.finish()


def eval_mean_std_5x500(config):
    _init_wandb(config, "eval")
    os.makedirs(config.save_dir, exist_ok=True)
    checkpoint_file = osp.join(config.save_dir, config.checkpoint_file)
    data_path = resolve_dataset_path(config.data_path)
    eval_source_path = resolve_dataset_path(getattr(config, "eval_data_path", None) or data_path)
    _wandb_update_config(
        {
            "resolved_data_path": data_path,
            "resolved_eval_source_path": eval_source_path,
            "checkpoint_file": checkpoint_file,
            "run_mode": "eval_mean_std",
        }
    )
    _wandb_log_event({"stage/run_start": 1})

    split_refs = _eval_split_refs(config, data_path)
    loaders, cache_path = _make_cache_loaders(
        config,
        split_refs,
        label="nspde_2d_eval_5x500",
        source_paths=[eval_source_path],
        shuffle_train=False,
    )

    _wandb_log_event({"stage/loaders_ready": 1, "cache_file_size_mb": os.path.getsize(cache_path) / (1024 ** 2)})
    model = _build_model(config)
    wandb.watch(model, log="gradients", log_freq=max(int(getattr(config, "print_every", 20)), 1))
    loss = LpLoss(size_average=False)
    model.load_state_dict(torch.load(checkpoint_file))
    _wandb_log_event({"stage/eval_start": 1})

    split_losses = []
    for split in ["test1", "test2", "test3", "test4", "test5"]:
        value = eval_nspde(model, loaders[split], loss, config.batch_size, device)
        print(f"{split}_loss:", value)
        split_losses.append((split, value))

    losses = [value for _, value in split_losses]
    mean_loss = statistics.mean(losses)
    std_loss = statistics.stdev(losses)
    print("mean_test_loss:", mean_loss)
    print("std_test_loss:", std_loss)
    _log_eval_mean_std_plot(split_losses, mean_loss, std_loss)
    _wandb_log_event({"stage/run_done": 1})
    wandb.finish()


def run_mode(config):
    mode = _mode(config)
    if mode in {"train", "train_1200"}:
        train_on_1200(config)
    elif mode in {"eval", "eval_mean_std", "eval_5x500"}:
        eval_mean_std_5x500(config)
    else:
        raise ValueError("mode must be one of train_1200 or eval_mean_std")


def hyperparameter_search(config):
    os.makedirs(config.save_dir, exist_ok=True)
    checkpoint_file = osp.join(config.save_dir, config.checkpoint_file)
    tmp_checkpoint_file = osp.join(config.save_dir, "tmp_checkpoint.pth")

    data_path = resolve_dataset_path(config.data_path)
    split_refs = split_train_val_test_refs(
        _training_refs(config, data_path),
        ntrain=config.ntrain,
        nval=config.nval,
        ntest=config.ntest,
    )
    loaders, _ = _make_cache_loaders(
        config,
        split_refs,
        label="nspde_2d_hparam",
        source_paths=[data_path],
        shuffle_train=True,
    )

    hyperparams = list(itertools.product(config.hidden_channels, config.n_iter, config.modes))
    loss = LpLoss(size_average=False)

    fieldnames = [
        "hidden_channels",
        "n_iter",
        "modes",
        "nb_params",
        "loss_train",
        "loss_val",
        "loss_test",
    ]
    log_file = osp.join(config.save_dir, config.log_file)
    with open(log_file, "w", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)

    best_loss_val = 1000.0
    for (_hidden_channels, _n_iter, _modes) in hyperparams:
        print(
            "\n hidden_channels:{}, n_iter:{}, modes:{}".format(
                _hidden_channels, _n_iter, _modes
            )
        )

        model = NeuralSPDE(
            dim=2,
            in_channels=1,
            noise_channels=1,
            hidden_channels=_hidden_channels,
            n_iter=_n_iter,
            modes1=_modes[0],
            modes2=_modes[1],
            modes3=_modes[2],
            solver=config.solver,
        ).cuda()

        nb_params = count_params(model)
        print("\n The model has {} parameters".format(nb_params))

        _, losses_train, losses_val = train_nspde(
            model,
            loaders["train"],
            loaders["val"],
            device,
            loss,
            batch_size=config.batch_size,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            plateau_patience=config.plateau_patience,
            factor=config.factor,
            delta=config.delta,
            plateau_terminate=config.plateau_terminate,
            print_every=config.print_every,
            checkpoint_file=tmp_checkpoint_file,
        )
        _log_training_loss_plots(losses_train, losses_val, config.print_every)

        model.load_state_dict(torch.load(tmp_checkpoint_file))
        loss_train = eval_nspde(model, loaders["train"], loss, config.batch_size, device)
        loss_val = eval_nspde(model, loaders["val"], loss, config.batch_size, device)
        loss_test = eval_nspde(model, loaders["test"], loss, config.batch_size, device)
        print("loss_train (model saved in tmp_checkpoint):", loss_train)
        print("loss_val (model saved in tmp_checkpoint):", loss_val)
        print("loss_test (model saved in tmp_checkpoint):", loss_test)
        if wandb.run is not None:
            _log_final_loss_plot(loss_train, loss_val, loss_test)

        if loss_val < best_loss_val:
            torch.save(model.state_dict(), checkpoint_file)
            best_loss_val = loss_val

        with open(log_file, "a", encoding="UTF8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [_hidden_channels, _n_iter, _modes, nb_params, loss_train, loss_val, loss_test]
            )

    print("Best model saved in:", checkpoint_file)


@hydra.main(version_base=None, config_path="../config/", config_name="nspde_phi42.yaml")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    seed = cfg.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    datasets = getattr(cfg, "datasets", None)
    configured_data_path = getattr(cfg, "data_path", None)
    if datasets:
        for dataset in datasets:
            cfg.data = dataset
            if configured_data_path:
                cfg.data_path = resolve_dataset_path(configured_data_path)
            else:
                cfg.data_path = resolve_dataset_path(
                    f"../../data_gen/results/{cfg.equation}/{dataset}"
                )
            path = dataset.replace("_3000", "_1200")
            cfg.checkpoint_file = f"{path}.pth"
            cfg.log_file = f"{dataset}_log"
            run_mode(cfg)
    else:
        cfg.data_path = resolve_dataset_path(cfg.data_path)
        run_mode(cfg)


if __name__ == "__main__":
    main()
