import scipy.io
import random
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import os.path as osp
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(osp.join(current_directory, "..",".."))
from model.NSPDE.utilities import *
from model.utilities import *
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_training(data_path, ntrain, ntest, batch_size, sub_x, T, sub_t,
                 hidden_channels, n_iter, modes1, modes2, solver,
                 epochs, learning_rate, scheduler_step, scheduler_gamma,
                 print_every, save_path):

    # Load data.
    data = scipy.io.loadmat(data_path)
    W, Sol = data['forcing'], data['sol']
    xi = torch.from_numpy(W.astype(np.float32))
    data = torch.from_numpy(Sol.astype(np.float32))

    train_loader, test_loader = dataloader_nspde_2d(u=data, xi=xi, ntrain=ntrain,
                                                    ntest=ntest, T=T, sub_t=sub_t, sub_x=sub_x,
                                                    batch_size=batch_size)

    # Define the model.
    model = NeuralSPDE(dim=2, in_channels=1, noise_channels=1, hidden_channels=hidden_channels,
                       n_iter=n_iter, modes1=modes1, modes2=modes2, solver=solver).cuda()

    print('The model has {} parameters'.format(count_params(model)))

    # Train the model.
    loss = LpLoss(size_average=False)
    model, losses_train, losses_test = train_nspde(model, train_loader, test_loader, device, loss,
                                                   batch_size=batch_size, epochs=epochs,
                                                   learning_rate=learning_rate, scheduler_step=scheduler_step,
                                                   scheduler_gamma=scheduler_gamma, print_every=print_every)

    torch.save(model.state_dict(), save_path)

def train(config):

    os.makedirs(config.save_dir, exist_ok=True)
    checkpoint_file = config.save_dir + config.checkpoint_file

    # Load data
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
    # indices = np.random.permutation(Sol.shape[0])
    # print('indices:', indices[:10])
    # Sol = Sol[indices]
    # W = W[indices]

    xi = torch.from_numpy(W.astype(np.float32))
    data = torch.from_numpy(Sol.astype(np.float32))
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

    model = NeuralSPDE(dim=2, in_channels=1, noise_channels=1, hidden_channels=config.hidden_channels,
                       n_iter=config.n_iter, modes1=config.modes[0], modes2=config.modes[1],
                       modes3=config.modes[2],
                       solver=config.solver).cuda()
    print('The model has {} parameters'.format(count_params(model)))

    loss = LpLoss(size_average=False)

    _, _, _ = train_nspde(model, train_loader, val_loader, device, loss,
                          batch_size=config.batch_size,
                          epochs=config.epochs,
                          learning_rate=config.learning_rate,
                          weight_decay=config.weight_decay,
                          plateau_patience=config.plateau_patience,
                          factor=config.factor,
                          plateau_terminate=config.plateau_terminate,
                          delta=config.delta,
                          print_every=config.print_every,
                          checkpoint_file=checkpoint_file)

    model.load_state_dict(torch.load(checkpoint_file))
    loss_train = eval_nspde(model, train_loader, loss, config.batch_size, device)
    loss_val = eval_nspde(model, val_loader, loss, config.batch_size, device)
    loss_test = eval_nspde(model, test_loader, loss, config.batch_size, device)
    print('loss_train (model saved in checkpoint):', loss_train)
    print('loss_val (model saved in checkpoint):', loss_val)
    print('loss_test (model saved in checkpoint):', loss_test)
    # plot_2d_u0xi(model, test_loader, device, T=config.T // config.sub_t)


def hyperparameter_tuning(data_path, ntrain, nval, ntest, batch_size, epochs, learning_rate,
                 plateau_patience, plateau_terminate, print_every,
                 sub_x, T, sub_t,
                 hidden_channels, n_iter, modes1, modes2, solver,
                 log_file, checkpoint_file, final_checkpoint_file):
    # Load data.
    data = scipy.io.loadmat(data_path)
    W, Sol = data['forcing'], data['sol']
    xi = torch.from_numpy(W.astype(np.float32))
    data = torch.from_numpy(Sol.astype(np.float32))

    _, test_dl = dataloader_nspde_2d(u=data, xi=xi, ntrain=ntrain+nval,
                                         ntest=ntest, T=T, sub_t=sub_t, sub_x=sub_x,
                                         batch_size=batch_size)

    train_dl, val_dl = dataloader_nspde_2d(u=data[:ntrain + nval], xi=xi[:ntrain + nval],
                                                   ntrain=ntrain, ntest=nval, T=T, sub_t=sub_t, sub_x=sub_x,
                                                   batch_size=batch_size)

    hyperparameter_search_nspde_2d(train_dl, val_dl, test_dl, solver=solver,
                                   d_h=hidden_channels, iter=n_iter, modes1=modes1, modes2=modes2,
                                   epochs=epochs, print_every=print_every, lr=learning_rate,
                                   plateau_patience=plateau_patience, plateau_terminate=plateau_terminate,
                                   log_file=log_file + '.csv', checkpoint_file=checkpoint_file,
                                   final_checkpoint_file=final_checkpoint_file)


@hydra.main(version_base=None, config_path="../config/", config_name="nspde_ns.yaml")
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
    # run_training(**cfg.args)
    # hyperparameter_tuning(**cfg.tuning)


if __name__ == '__main__':
    main()