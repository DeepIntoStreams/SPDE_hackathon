import scipy.io
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import os.path as osp
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(osp.join(current_directory, "..",".."))
from model.NCDE.NCDE import *
from model.utilities import *
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(data_path, ntrain, ntest, batch_size, epochs, learning_rate,
                 scheduler_step, scheduler_gamma, print_every,
                 dim_x, T, sub_t,
                 normalizer, interpolation,
                 hidden_channels, solver, save_path):

    data = scipy.io.loadmat(data_path)

    O_X, O_T, W, Sol = data['X'], data['T'], data['W'], data['sol']
    xi = torch.from_numpy(W.astype(np.float32))
    data = torch.from_numpy(Sol.astype(np.float32))

    train_loader, test_loader, normalizer = dataloader_ncde_1d(u=data, xi=xi, ntrain=ntrain,
                                                               ntest=ntest, T=T, sub_t=sub_t,
                                                               batch_size=batch_size, dim_x=dim_x,
                                                               normalizer=normalizer, interpolation=interpolation)

    model = NeuralCDE(input_channels=dim_x+1, hidden_channels=hidden_channels, output_channels=dim_x,
                      interpolation=interpolation, solver=solver).cuda()

    print('The model has {} parameters'. format(count_params(model)))

    loss = LpLoss(size_average=False)

    model, losses_train, losses_test = train_ncde(model, train_loader, test_loader, normalizer,
                                                    device, loss, batch_size=batch_size, epochs=epochs,
                                                    learning_rate=learning_rate, scheduler_step=scheduler_step,
                                                    scheduler_gamma=scheduler_gamma, print_every=print_every)

    torch.save(model.state_dict(), save_path)

def hyperparameter_tuning(data_path, ntrain, nval, ntest, batch_size, epochs, learning_rate,
                 plateau_patience, plateau_terminate, print_every,
                 dim_x, T, sub_t,
                 normalizer, interpolation, hidden_channels, solver,
                 log_file, checkpoint_file, final_checkpoint_file):

    data = scipy.io.loadmat(data_path)

    O_X, O_T, W, Sol = data['X'], data['T'], data['W'], data['sol']
    xi = torch.from_numpy(W.astype(np.float32))
    data = torch.from_numpy(Sol.astype(np.float32))

    _, test_dl, norm = dataloader_ncde_1d(u=data, xi=xi, ntrain=ntrain + nval,
                                          ntest=ntest, T=T, sub_t=sub_t, normalizer=normalizer,
                                          batch_size=batch_size, dim_x=dim_x, interpolation=interpolation)

    train_dl, val_dl, norm = dataloader_ncde_1d(u=data[:ntrain + nval], xi=xi[:ntrain + nval],
                                                ntrain=ntrain, ntest=nval, T=T, sub_t=sub_t, normalizer=normalizer,
                                                batch_size=batch_size, dim_x=dim_x, interpolation=interpolation)

    hyperparameter_search_ncde(train_dl, val_dl, test_dl, dim_x, u_normalizer=norm, d_h=hidden_channels, solver=solver,
                               epochs=epochs, print_every=print_every, lr=learning_rate, plateau_patience=plateau_patience, plateau_terminate=plateau_terminate,
                               log_file=log_file + '.csv', checkpoint_file=checkpoint_file, final_checkpoint_file=final_checkpoint_file)



@hydra.main(version_base=None, config_path="../config/", config_name="ncde")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    train(**cfg.args)
    # hyperparameter_tuning(**cfg.tuning)


if __name__ == '__main__':
    main()

