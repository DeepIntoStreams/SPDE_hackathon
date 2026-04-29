import scipy.io
import random
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import os
import os.path as osp
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
main_root = osp.join(current_directory, "..", "..")
if main_root not in sys.path:
    sys.path.insert(0, main_root)
from model.NSPDE.utilities import *
from model.utilities import *
from evaluations.metrics import (
    AutoCorrelationMetric,
    CrossCorrelationMetric,
    HsLossMetric,
    LpLossMetric,
)
import warnings
warnings.filterwarnings('ignore')

try:
    import wandb
except ImportError:
    wandb = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_seed_list(config):
    seeds = getattr(config, "seeds", None)
    if seeds is None:
        return [int(config.seed)]
    return [int(seed) for seed in seeds]


def append_seed_to_name(name, seed):
    if name is None:
        return f"seed{seed}"
    name = str(name)
    if f"seed{seed}" in name:
        return name
    root, ext = osp.splitext(name)
    if ext:
        return f"{root}_seed{seed}{ext}"
    return f"{name}_seed{seed}"


def make_seed_config(config, seed):
    seed_config = copy.deepcopy(config)
    base_save_dir = str(getattr(config, "save_dir", "."))
    base_wandb_name = getattr(config, "wandb_name", None)
    base_best_ckpt_name = getattr(config, "best_ckpt_name", "nspde_phi42_best_val.pt")
    with open_dict(seed_config):
        seed_config.seed = int(seed)
        seed_config.save_dir = osp.join(base_save_dir, f"seed_{seed}")
        seed_config.wandb_name = append_seed_to_name(base_wandb_name, seed)
        seed_config.best_ckpt_name = append_seed_to_name(base_best_ckpt_name, seed)
    return seed_config


def print_seed_summary(seed_results):
    if len(seed_results) <= 1:
        return

    metric_keys = [
        "loss_train", "loss_val", "loss_test",
        "test_Rel_L2", "test_W1_2", "test_AutoCorr", "test_CrossCorr",
        "inference_avg_sample_time", "inference_throughput",
    ]
    print("\nMulti-seed summary (mean +/- std):")
    for key in metric_keys:
        values = [result[key] for result in seed_results if result.get(key) is not None]
        if len(values) == 0:
            continue
        values = np.asarray(values, dtype=np.float64)
        print(f"{key}: {values.mean():.6f} +/- {values.std(ddof=1):.6f}")


def use_wandb(config):
    return bool(getattr(config, "use_wandb", False))


def init_wandb(config, default_project):
    if not use_wandb(config):
        return None
    if wandb is None:
        raise ImportError("use_wandb=true but wandb is not installed. Run `pip install wandb` first.")

    run = wandb.init(
        entity=getattr(config, "wandb_entity", None),
        project=getattr(config, "wandb_project", default_project),
        name=getattr(config, "wandb_name", None),
        config=OmegaConf.to_container(config, resolve=True),
    )
    return run


def wandb_log(data, step=None):
    if wandb is not None and wandb.run is not None:
        clean_data = {key: value for key, value in data.items() if value is not None}
        wandb.log(clean_data, step=step)


def wandb_log_file(path, key=None, as_image=False):
    if wandb is None or wandb.run is None or path is None or not osp.exists(path):
        return
    if as_image and key is not None:
        wandb.log({key: wandb.Image(path)})
    wandb.save(path)


def finish_wandb():
    if wandb is not None and wandb.run is not None:
        wandb.finish()


def load_best_checkpoint(model, checkpoint_path, device):
    if checkpoint_path is None or not osp.exists(checkpoint_path):
        print("best checkpoint not found; using current model for final metrics")
        return None
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    best_epoch = checkpoint.get("epoch", None) if isinstance(checkpoint, dict) else None
    best_val_loss = checkpoint.get("best_val_loss", None) if isinstance(checkpoint, dict) else None
    print(f"loaded best checkpoint: {checkpoint_path}")
    if best_epoch is not None and best_val_loss is not None:
        print(f"best checkpoint epoch: {best_epoch} | best val loss: {best_val_loss:.6f}")
    return checkpoint


def collect_nspde_predictions(model, data_loader, device):
    model.eval()
    reals, preds = [], []
    with torch.no_grad():
        for u0_, xi_, u_ in data_loader:
            u0_ = u0_.to(device)
            xi_ = xi_.to(device)
            u_pred = model(u0_, xi_)
            if u_pred.ndim == 5 and u_pred.shape[1] == 1:
                u_pred = u_pred[:, 0]
            preds.append(u_pred.detach().cpu())
            reals.append(u_.detach().cpu())
    return torch.cat(reals, dim=0), torch.cat(preds, dim=0)


def to_path_tensor(u):
    if u.ndim == 5 and u.shape[1] == 1:
        u = u[:, 0]
    if u.ndim != 4:
        raise ValueError(f"Expected [B, X, Y, T] tensor for path metrics, got shape={tuple(u.shape)}")
    b, x, y, t = u.shape
    return u.permute(0, 3, 1, 2).reshape(b, t, x * y)


def evaluate_phi42_metrics(model, data_loader, device, prefix="test", include_rel_l2=True, verbose=True):
    u_real, u_pred = collect_nspde_predictions(model, data_loader, device)
    u_real_path = to_path_tensor(u_real)
    u_pred_path = to_path_tensor(u_pred)

    metric_specs = []
    if include_rel_l2:
        metric_specs.append(("Rel_L2", LpLossMetric(mode='rel'), u_real, u_pred))
    metric_specs.extend([
        ("W1_2", HsLossMetric(k=1), u_real, u_pred),
        ("AutoCorr", AutoCorrelationMetric(), u_real_path, u_pred_path),
        ("CrossCorr", CrossCorrelationMetric(), u_real_path, u_pred_path),
    ])

    if verbose:
        print(f"{prefix} metrics:")
    results = {}
    for name, metric, real, pred in metric_specs:
        try:
            value = metric.measure(real, pred)
            results[name] = float(value)
            if verbose:
                print(f"{prefix}_{name}: {results[name]:.6f}")
        except (RuntimeError, ValueError) as err:
            results[name] = None
            if verbose:
                print(f"{prefix}_{name}: failed ({err})")
    return results


def plot_loss_curve(loss_epochs, train_losses, val_losses, save_dir):
    if len(train_losses) == 0 or len(val_losses) == 0:
        return None
    plt.figure(figsize=(9, 6))
    plt.plot(loss_epochs, train_losses, marker='o', markersize=7, linewidth=3, label='train L2')
    plt.plot(loss_epochs, val_losses, marker='o', markersize=7, linewidth=3, label='val L2')
    plt.title('NSPDE Phi42 L2 Loss', fontsize=22, fontweight='bold')
    plt.xlabel('Epoch', fontsize=18, fontweight='bold')
    plt.ylabel('Relative L2 loss', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=15, frameon=True)
    plt.grid(True, alpha=0.35, linewidth=1.2)
    plt.tight_layout()
    out_path = osp.join(save_dir, "loss_curve.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"saved loss curve: {out_path}")
    return out_path


def plot_metric_curve(metric_history, save_dir):
    if len(metric_history) == 0:
        return None
    metric_names = [name for name in ("W1_2", "AutoCorr", "CrossCorr") if any(row.get(name) is not None for row in metric_history)]
    if len(metric_names) == 0:
        return None

    plt.figure(figsize=(9, 6))
    for name in metric_names:
        xs, ys = [], []
        for row in metric_history:
            value = row.get(name)
            if value is not None:
                xs.append(row["epoch"])
                ys.append(value)
        if len(xs) > 0:
            plt.plot(xs, ys, marker='o', markersize=7, linewidth=3, label=name)
    plt.title('NSPDE Phi42 Validation Metrics', fontsize=22, fontweight='bold')
    plt.xlabel('Epoch', fontsize=18, fontweight='bold')
    plt.ylabel('Validation metric', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=15, frameon=True)
    plt.grid(True, alpha=0.35, linewidth=1.2)
    plt.tight_layout()
    out_path = osp.join(save_dir, "metric_curve.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"saved metric curve: {out_path}")
    return out_path


def load_phi42_data(config):
    data = scipy.io.loadmat(config.data_path)
    if config.equation == 'phi42':
        W, Sol = data['W'], data['sol']
        W = np.transpose(W, (0, 2, 3, 1))
        Sol = np.transpose(Sol, (0, 2, 3, 1))
    elif config.equation == 'NS':
        W, Sol = data['forcing'], data['sol']
    else:
        print('Unknown equation')
        exit(0)
    print('data shape:')
    print(W.shape)
    print(Sol.shape)

    indices = np.random.permutation(Sol.shape[0])
    print('indices:', indices[:10])
    Sol = Sol[indices]
    W = W[indices]

    xi = torch.from_numpy(W.astype(np.float32))
    data = torch.from_numpy(Sol.astype(np.float32))
    return data, xi


def make_loaders(data, xi, config):
    ntrain, nval, ntest = config.ntrain, config.nval, config.ntest
    _, test_loader = dataloader_nspde_2d(u=data, xi=xi,
                                         ntrain=ntrain + nval,
                                         ntest=ntest,
                                         T=config.T,
                                         sub_t=config.sub_t,
                                         sub_x=config.sub_x,
                                         batch_size=config.batch_size)
    train_loader, val_loader = dataloader_nspde_2d(u=data[:ntrain + nval], xi=xi[:ntrain + nval],
                                                   ntrain=ntrain,
                                                   ntest=nval,
                                                   T=config.T,
                                                   sub_t=config.sub_t,
                                                   sub_x=config.sub_x,
                                                   batch_size=config.batch_size)
    return train_loader, val_loader, test_loader


def train(config):

    os.makedirs(config.save_dir, exist_ok=True)
    init_wandb(config, default_project="NSPDE_Phi42")
    data, xi = load_phi42_data(config)
    train_loader, val_loader, test_loader = make_loaders(data, xi, config)
    metric_eval_every = int(getattr(config, "metric_eval_every", 50))

    model = NeuralSPDE(dim=2, in_channels=1, noise_channels=1, hidden_channels=config.hidden_channels,
                       n_iter=config.n_iter, modes1=config.modes[0], modes2=config.modes[1],
                       modes3=config.modes[2],
                       solver=config.solver).cuda()
    print('The model has {} parameters'.format(count_params(model)))

    loss = LpLoss(size_average=False)
    metric_history = []
    best_ckpt_name = getattr(config, "best_ckpt_name", "nspde_phi42_best_val.pt")
    best_ckpt_path = osp.join(config.save_dir, best_ckpt_name)

    def metric_eval_callback(ep, current_model):
        metrics = evaluate_phi42_metrics(
            current_model, val_loader, device,
            prefix=f"val_epoch_{ep}",
            include_rel_l2=False,
            verbose=True,
        )
        metrics["epoch"] = ep
        metric_history.append(metrics)
        wandb_log({"epoch": ep, **{f"val/{key}": value for key, value in metrics.items() if key != "epoch"}})

    def epoch_log_callback(ep, train_loss, val_loss, test_loss):
        wandb_log({
            "epoch": ep,
            "loss/train": train_loss,
            "loss/val": val_loss,
            "loss/test": test_loss,
        })

    model, losses_train, losses_val = train_nspde(
        model, train_loader, val_loader, device, loss,
        batch_size=config.batch_size,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        plateau_patience=config.plateau_patience,
        factor=config.factor,
        plateau_terminate=config.plateau_terminate,
        delta=config.delta,
        print_every=config.print_every,
        report_loader=test_loader,
        report_name='Test',
        metric_eval_fn=metric_eval_callback,
        metric_eval_every=metric_eval_every,
        epoch_log_fn=epoch_log_callback,
        checkpoint_file=best_ckpt_path,
    )

    loss_epochs = [i * config.print_every for i in range(len(losses_train))]
    loss_curve_path = plot_loss_curve(loss_epochs, losses_train, losses_val, config.save_dir)
    metric_curve_path = plot_metric_curve(metric_history, config.save_dir)
    wandb_log_file(loss_curve_path, key="plots/loss_curve", as_image=True)
    wandb_log_file(metric_curve_path, key="plots/metric_curve", as_image=True)

    best_checkpoint = load_best_checkpoint(model, best_ckpt_path, device)
    wandb_log_file(best_ckpt_path)

    loss_train = eval_nspde(model, train_loader, loss, config.batch_size, device)
    loss_val = eval_nspde(model, val_loader, loss, config.batch_size, device)
    loss_test = eval_nspde(model, test_loader, loss, config.batch_size, device)
    inference_stats = measure_inference_time_nspde(model, test_loader, device)
    print('final loss_train:', loss_train)
    print('final loss_val:', loss_val)
    print('final loss_test:', loss_test)
    print('inference_total_time (test loader, warmup excluded):', inference_stats['total_time'])
    print('inference_avg_batch_time:', inference_stats['avg_batch_time'])
    print('inference_avg_sample_time:', inference_stats['avg_sample_time'])
    print('inference_throughput_samples_per_sec:', inference_stats['throughput'])
    test_metrics = evaluate_phi42_metrics(model, test_loader, device, prefix="test", include_rel_l2=True, verbose=True)
    final_results = {
        "seed": int(getattr(config, "seed", -1)),
        "loss_train": loss_train,
        "loss_val": loss_val,
        "loss_test": loss_test,
        "best_checkpoint_epoch": best_checkpoint.get("epoch") if isinstance(best_checkpoint, dict) else None,
        "best_checkpoint_val_loss": best_checkpoint.get("best_val_loss") if isinstance(best_checkpoint, dict) else None,
        **{f"inference_{key}": value for key, value in inference_stats.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    wandb_log({
        "final/loss_train": loss_train,
        "final/loss_val": loss_val,
        "final/loss_test": loss_test,
        "best/epoch": best_checkpoint.get("epoch") if isinstance(best_checkpoint, dict) else None,
        "best/val_loss": best_checkpoint.get("best_val_loss") if isinstance(best_checkpoint, dict) else None,
        **{f"inference/{key}": value for key, value in inference_stats.items()},
        **{f"test/{key}": value for key, value in test_metrics.items()},
    })
    finish_wandb()
    return final_results
    # plot_2d_u0xi(model, test_loader, device, T=config.T // config.sub_t)


def hyperparameter_search(config):
    os.makedirs(config.save_dir, exist_ok=True)
    data, xi = load_phi42_data(config)
    train_loader, val_loader, test_loader = make_loaders(data, xi, config)
    metric_eval_every = int(getattr(config, "metric_eval_every", 50))
    hyperparams = list(itertools.product(config.hidden_channels, config.n_iter, config.modes))

    loss = LpLoss(size_average=False)

    fieldnames = ['hidden_channels', 'n_iter', 'modes', 'nb_params', 'loss_train', 'loss_val', 'loss_test']
    log_file = config.save_dir + config.log_file
    with open(log_file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)

    best_loss_val = 1000.
    best_state_dict = None

    for (_hidden_channels, _n_iter, _modes) in hyperparams:

        print('\n hidden_channels:{}, n_iter:{}, modes:{}'.format(_hidden_channels, _n_iter, _modes))

        model = NeuralSPDE(dim=2, in_channels=1, noise_channels=1, hidden_channels=_hidden_channels,
                           n_iter=_n_iter, modes1=_modes[0], modes2=_modes[1], modes3=_modes[2],
                           solver=config.solver).cuda()

        nb_params = count_params(model)

        print('\n The model has {} parameters'.format(nb_params))

        metric_history = []

        def metric_eval_callback(ep, current_model):
            metrics = evaluate_phi42_metrics(
                current_model, val_loader, device,
                prefix=f"val_epoch_{ep}",
                include_rel_l2=False,
                verbose=True,
            )
            metrics["epoch"] = ep
            metric_history.append(metrics)

        model, losses_train, losses_val = train_nspde(
            model, train_loader, val_loader, device, loss,
            batch_size=config.batch_size,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            plateau_patience=config.plateau_patience,
            factor=config.factor,
            delta=config.delta,
            plateau_terminate=config.plateau_terminate,
            print_every=config.print_every,
            report_loader=test_loader,
            report_name='Test',
            metric_eval_fn=metric_eval_callback,
            metric_eval_every=metric_eval_every,
        )
        loss_epochs = [i * config.print_every for i in range(len(losses_train))]
        plot_loss_curve(loss_epochs, losses_train, losses_val, config.save_dir)
        plot_metric_curve(metric_history, config.save_dir)
        loss_train = eval_nspde(model, train_loader, loss, config.batch_size, device)
        loss_val = eval_nspde(model, val_loader, loss, config.batch_size, device)
        loss_test = eval_nspde(model, test_loader, loss, config.batch_size, device)
        inference_stats = measure_inference_time_nspde(model, test_loader, device)
        print('final loss_train:', loss_train)
        print('final loss_val:', loss_val)
        print('final loss_test:', loss_test)
        print('inference_total_time (test loader, warmup excluded):', inference_stats['total_time'])
        print('inference_avg_batch_time:', inference_stats['avg_batch_time'])
        print('inference_avg_sample_time:', inference_stats['avg_sample_time'])
        print('inference_throughput_samples_per_sec:', inference_stats['throughput'])
        evaluate_phi42_metrics(model, test_loader, device, prefix="test", include_rel_l2=True, verbose=True)

        # keep the best model in memory, without checkpoint files
        if loss_val < best_loss_val:
            best_loss_val = loss_val
            best_state_dict = copy.deepcopy(model.state_dict())

        # write results
        with open(log_file, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([_hidden_channels, _n_iter, _modes, nb_params, loss_train, loss_val, loss_test])

    if best_state_dict is not None:
        print('Best model selected in memory with val loss:', best_loss_val)


@hydra.main(version_base=None, config_path="../config/", config_name="nspde_ns.yaml")
def main(cfg: DictConfig):

    # print(OmegaConf.to_yaml(cfg, resolve=True))

    seed_results = []
    for seed in get_seed_list(cfg):
        print(f"\n========== NSPDE seed {seed} ==========")
        seed_cfg = make_seed_config(cfg, seed)
        set_random_seed(seed)
        seed_results.append(train(seed_cfg))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_seed_summary(seed_results)
    # hyperparameter_search(cfg)


if __name__ == '__main__':
    main()
