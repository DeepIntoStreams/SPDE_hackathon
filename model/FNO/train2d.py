import scipy.io
import random
import itertools
import csv
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import os.path as osp
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(osp.join(current_directory, "..", ".."))
from FNO2D import *
from model.utilities import *
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    print('W shape:', W.shape)
    print('Sol shape:', Sol.shape)

    indices = np.random.permutation(Sol.shape[0])
    print('indices:', indices[:10])
    Sol = Sol[indices]
    W = W[indices]

    xi = torch.from_numpy(W.astype(np.float32))
    data = torch.from_numpy(Sol.astype(np.float32))
    ntrain, nval, ntest = config.ntrain, config.nval, config.ntest

    _, test_loader = dataloader_fno_2d_xi(u=data, xi=xi,
                                          ntrain=ntrain + nval, ntest=ntest,
                                          T=config.T, sub_t=config.sub_t,
                                          sub_x=config.sub_x,
                                          batch_size=config.batch_size)
    train_loader, val_loader = dataloader_fno_2d_xi(u=data[:ntrain + nval], xi=xi[:ntrain + nval],
                                                    ntrain=ntrain, ntest=nval,
                                                    T=config.T, sub_t=config.sub_t,
                                                    sub_x=config.sub_x,
                                                    batch_size=config.batch_size)

    model = FNO_space2D_time(modes1=config.modes1, modes2=config.modes2, modes3=config.modes3, width=config.width,
                             L=config.L, T=config.T // config.sub_t).cuda()
    print('The model has {} parameters'.format(count_params(model)))

    loss = LpLoss(size_average=False)

    _, _, _ = train_fno_2d(model, train_loader, val_loader, device, loss,
                           batch_size=config.batch_size,
                           epochs=config.epochs,
                           learning_rate=config.learning_rate,
                           plateau_patience=config.plateau_patience,
                           delta=config.delta,
                           plateau_terminate=config.plateau_terminate,
                           print_every=config.print_every,
                           checkpoint_file=checkpoint_file)

    model.load_state_dict(torch.load(checkpoint_file))
    loss_train = eval_fno_2d(model, train_loader, loss, config.batch_size, device)
    loss_val = eval_fno_2d(model, val_loader, loss, config.batch_size, device)
    loss_test = eval_fno_2d(model, test_loader, loss, config.batch_size, device)
    print('loss_train (model saved in checkpoint):', loss_train)
    print('loss_val (model saved in checkpoint):', loss_val)
    print('loss_test (model saved in checkpoint):', loss_test)


def hyperparameter_search(config):
    os.makedirs(config.save_dir, exist_ok=True)
    checkpoint_file = config.save_dir + config.checkpoint_file
    tmp_checkpoint_file = config.save_dir + 'tmp_checkpoint.pth'

    # Load data
    data = scipy.io.loadmat(config.data_path)
    if config.equation == 'phi42':
        W, Sol = data['W'], data['sol']
        W = np.transpose(W, (0, 2, 3, 1))
        Sol = np.transpose(Sol, (0, 2, 3, 1))
    elif config.equation == 'NS':
        W, Sol = data['forcing'], data['sol']
    print('W shape:', W.shape)
    print('Sol shape:', Sol.shape)

    indices = np.random.permutation(Sol.shape[0])
    print('indices:', indices[:10])
    Sol = Sol[indices]
    W = W[indices]

    xi = torch.from_numpy(W.astype(np.float32))
    data = torch.from_numpy(Sol.astype(np.float32))
    ntrain, nval, ntest = config.ntrain, config.nval, config.ntest

    _, test_loader = dataloader_fno_2d_xi(u=data, xi=xi,
                                          ntrain=ntrain + nval, ntest=ntest,
                                          T=config.T, sub_t=config.sub_t,
                                          sub_x=config.sub_x,
                                          batch_size=config.batch_size)
    train_loader, val_loader = dataloader_fno_2d_xi(u=data[:ntrain + nval], xi=xi[:ntrain + nval],
                                                    ntrain=ntrain, ntest=nval,
                                                    T=config.T, sub_t=config.sub_t,
                                                    sub_x=config.sub_x,
                                                    batch_size=config.batch_size)

    hyperparams = list(itertools.product(config.width, config.L, config.modes1))

    loss = LpLoss(size_average=False)

    fieldnames = ['width', 'L', 'modes1', 'nb_params', 'loss_train', 'loss_val', 'loss_test']
    log_file = config.save_dir + config.log_file
    with open(log_file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)

    best_loss_val = 1000.

    for (_width, _L, _modes1) in hyperparams:

        print('\n width:{}, L:{}, modes1:{}'.format(_width, _L, _modes1))

        model = FNO_space2D_time(modes1=_modes1, modes2=config.modes2, modes3=config.modes3, width=_width,
                                 L=_L, T=config.T // config.sub_t).cuda()
        nb_params = count_params(model)

        print('\n The model has {} parameters'.format(nb_params))

        # Train the model. The best model is checkpointed.
        _, _, _ = train_fno_2d(model, train_loader, val_loader, device, loss,
                               batch_size=config.batch_size,
                               epochs=config.epochs,
                               learning_rate=config.learning_rate,
                               plateau_patience=config.plateau_patience,
                               delta=config.delta,
                               plateau_terminate=config.plateau_terminate,
                               print_every=config.print_every,
                               checkpoint_file=tmp_checkpoint_file)
        # load the best trained model
        model.load_state_dict(torch.load(tmp_checkpoint_file))
        loss_train = eval_fno_2d(model, train_loader, loss, config.batch_size, device)
        loss_val = eval_fno_2d(model, val_loader, loss, config.batch_size, device)
        loss_test = eval_fno_2d(model, test_loader, loss, config.batch_size, device)
        print('loss_train (model saved in tmp_checkpoint):', loss_train)
        print('loss_val (model saved in tmp_checkpoint):', loss_val)
        print('loss_test (model saved in tmp_checkpoint):', loss_test)

        # if this configuration of hyperparameters is the best so far (determined wihtout using the test set), save it
        if loss_val < best_loss_val:
            torch.save(model.state_dict(), checkpoint_file)
            best_loss_val = loss_val

        # write results
        with open(log_file, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([_width, _L, _modes1, nb_params, loss_train, loss_val, loss_test])

    print('Best model saved in:', checkpoint_file)


@hydra.main(version_base=None, config_path="../config/", config_name="fno_ns")
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
    # hyperparameter_search(cfg)


if __name__ == '__main__':
    main()
