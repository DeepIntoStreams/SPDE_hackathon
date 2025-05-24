import scipy.io
import random
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import os.path as osp
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(osp.join(current_directory, "..",".."))
from model.DeepONet.deepOnet import *
from model.utilities import *
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(config):
    os.makedirs(config.save_dir, exist_ok=True)
    checkpoint_file = config.save_dir + config.checkpoint_file

    # Load data
    data = scipy.io.loadmat(config.data_path)
    W, Sol = data['W'], data['sol']
    print('W shape:', W.shape)
    print('Sol shape:', Sol.shape)
    # indices = np.random.permutation(Sol.shape[0])
    # print('indices:', indices[:10])
    # Sol = Sol[indices]
    # W = W[indices]

    xi = torch.from_numpy(W.astype(np.float32))
    data = torch.from_numpy(Sol.astype(np.float32))

    ntrain, nval, ntest = config.ntrain, config.nval, config.ntest

    train_loader, test_loader, norm, grid = dataloader_deeponet_1d_xi(u=data, xi=xi,
                                                                      ntrain=ntrain,
                                                                      ntest=ntest,
                                                                      T=config.T,
                                                                      sub_t=config.sub_t,
                                                                      batch_size=config.batch_size,
                                                                      dim_x=config.dim_x,
                                                                      normalizer=config.normalizer)
    _, val_loader, _, _ = dataloader_deeponet_1d_xi(u=data[:ntrain + nval],
                                                    xi=xi[:ntrain + nval],
                                                    ntrain=ntrain,
                                                    ntest=nval,
                                                    T=config.T,
                                                    sub_t=config.sub_t,
                                                    batch_size=config.batch_size,
                                                    dim_x=config.dim_x,
                                                    normalizer=config.normalizer)

    S = config.dim_x * (config.T - 1)
    branch = [S] + config.branch_depth * [config.width]
    trunk = [grid.shape[-1]] + config.trunk_depth * [config.width]
    model = DeepONetCP(branch_layer=branch,
                       trunk_layer=trunk).to(device)

    print('The model has {} parameters'.format(count_params(model)))

    loss = LpLoss(size_average=False)

    _, _, _ = train_deepOnet_1d(model, train_loader, val_loader,
                                grid, norm, device, loss,
                                batch_size=config.batch_size,
                                epochs=config.epochs,
                                learning_rate=config.learning_rate,
                                plateau_patience=config.plateau_patience,
                                plateau_terminate=config.plateau_terminate,
                                delta=config.delta,
                                print_every=config.print_every,
                                checkpoint_file=checkpoint_file)

    model.load_state_dict(torch.load(checkpoint_file))
    loss_train = eval_deeponet(model, train_loader, loss, config.batch_size, device, grid, norm)
    loss_val = eval_deeponet(model, val_loader, loss, config.batch_size, device, grid, norm)
    loss_test = eval_deeponet(model, test_loader, loss, config.batch_size, device, grid, norm)
    print('loss_train (model saved in checkpoint):', loss_train)
    print('loss_val (model saved in checkpoint):', loss_val)
    print('loss_test (model saved in checkpoint):', loss_test)


def hyperparameter_tuning(data_path, ntrain, nval, ntest, batch_size, epochs, learning_rate,
                 plateau_patience, plateau_terminate, print_every,
                 dim_x, T, sub_t, normalizer,
                 width, branch_depth, trunk_depth,
                 log_file, checkpoint_file, final_checkpoint_file):

    data = scipy.io.loadmat(data_path)
    O_X, O_T, W, Sol = data['X'], data['T'], data['W'], data['sol']
    xi = torch.from_numpy(W.astype(np.float32))
    data = torch.from_numpy(Sol.astype(np.float32))
    _, test_dl, norm, grid = dataloader_deeponet_1d_xi(u=data, xi=xi, ntrain=ntrain + nval,
                                                       ntest=ntest, T=T, sub_t=sub_t,
                                                       batch_size=batch_size, dim_x=dim_x, normalizer=normalizer)

    train_dl, val_dl, norm, grid = dataloader_deeponet_1d_xi(u=data[:ntrain + nval], xi=xi[:ntrain + nval],
                                                             ntrain=ntrain, ntest=nval, T=T, sub_t=sub_t,
                                                             batch_size=batch_size, dim_x=dim_x, normalizer=normalizer)

    hyperparameter_search_deeponet(train_dl, val_dl, test_dl, dim_x * (T - 1), grid, norm,
                                   width=width, branch_depth=branch_depth, trunk_depth=trunk_depth,
                                   lr=learning_rate, epochs=epochs, print_every=print_every, plateau_patience=plateau_patience,
                                   plateau_terminate=plateau_terminate, log_file=log_file + '.csv',
                                   checkpoint_file=checkpoint_file,
                                   final_checkpoint_file=final_checkpoint_file)


@hydra.main(version_base=None, config_path="../config/", config_name="deeponet")
def main(cfg: DictConfig):

    # print(OmegaConf.to_yaml(cfg, resolve=True))

    # Set random seed
    seed = cfg.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train(cfg)
    # hyperparameter_tuning(**cfg.tuning)


if __name__ == '__main__':
    main()