import scipy.io
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import os.path as osp
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(osp.join(current_directory, "..",".."))
from FNO2D import *
from model.utilities import *
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_training_xi(data_path, ntrain, ntest, batch_size, epochs, learning_rate,
                    scheduler_step, scheduler_gamma, print_every,
                    sub_x, T, sub_t, modes1, modes2, modes3, width, L,
                    save_path):
    # Load data.
    data = scipy.io.loadmat(data_path)
    W, Sol = data['forcing'], data['sol']
    xi = torch.from_numpy(W.astype(np.float32))
    data = torch.from_numpy(Sol.astype(np.float32))

    train_loader, test_loader = dataloader_fno_2d_xi(u=data, xi=xi, ntrain=ntrain,
                                                    ntest=ntest, T=T, sub_t=sub_t, sub_x=sub_x,
                                                    batch_size=batch_size)

    model = FNO_space2D_time(modes1, modes2, modes3, width, L, T=T//sub_t).cuda()

    print('The model has {} parameters'. format(count_params(model)))

    loss = LpLoss(size_average=False)

    model, losses_train, losses_test = train_fno_2d(model, train_loader, test_loader,
                                                    device, loss, batch_size=batch_size, epochs=epochs,
                                                    learning_rate=learning_rate, scheduler_step=scheduler_step,
                                                    scheduler_gamma=scheduler_gamma, print_every=print_every)

    torch.save(model.state_dict(), save_path)
    plot_2d_xi(model, test_loader, device)

def run_training_u0(data_path, ntrain, ntest, batch_size, epochs, learning_rate,
                    scheduler_step, scheduler_gamma, print_every,
                    sub_x, T, sub_t,
                    modes1, modes2, modes3, width, L,
                    save_path):
    # Load data.
    data = scipy.io.loadmat(data_path)
    Sol = data['sol']
    data = torch.from_numpy(Sol.astype(np.float32))

    train_loader, test_loader = dataloader_fno_2d_u0(u=data, ntrain=ntrain, ntest=ntest,
                                                     T=T, sub_t=sub_t, sub_x=sub_x,
                                                     batch_size=batch_size)

    model = FNO_space2D_time(modes1, modes2, modes3, width, L, T=T // sub_t).cuda()

    print('The model has {} parameters'.format(count_params(model)))

    loss = LpLoss(size_average=False)

    model, losses_train, losses_test = train_fno_2d(model, train_loader, test_loader,
                                                    device, loss, batch_size=batch_size, epochs=epochs,
                                                    learning_rate=learning_rate, scheduler_step=scheduler_step,
                                                    scheduler_gamma=scheduler_gamma, print_every=print_every)

    torch.save(model.state_dict(), save_path)


@hydra.main(version_base=None, config_path="../config/", config_name="fno_NS")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    run_training_xi(**cfg.args)
    # run_training_u0(**cfg.args)



if __name__ == '__main__':
    main()
