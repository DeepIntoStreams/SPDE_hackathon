import scipy.io
import random
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import os.path as osp
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(osp.join(current_directory, "..",".."))
from WNO1D import *
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
    xi = torch.from_numpy(W.astype(np.float32))
    data = torch.from_numpy(Sol.astype(np.float32))

    ntrain, nval, ntest = config.ntrain, config.nval, config.ntest

    _, test_loader = dataloader_wno_1d_xi(u=data, xi=xi,
                                          ntrain=ntrain+nval,
                                          ntest=ntest,
                                          T=config.T,
                                          sub_t=config.sub_t,
                                          batch_size=config.batch_size,
                                          dim_x=config.dim_x)
    train_loader, val_loader = dataloader_wno_1d_xi(u=data[:ntrain + nval], xi=xi[:ntrain + nval],
                                                    ntrain=ntrain,
                                                    ntest=nval,
                                                    T=config.T,
                                                    sub_t=config.sub_t,
                                                    batch_size=config.batch_size,
                                                    dim_x=config.dim_x)

    model = WNO_space1D_time(level=config.level, width=config.width, L=config.L, T=config.T // config.sub_t,
                             input_sample=xi[:1, :config.dim_x, 0:config.T:config.sub_t]).cuda()
    print('The model has {} parameters'.format(count_params(model)))

    loss = LpLoss(size_average=False)

    _, _, _ = train_wno_1d(model, train_loader, val_loader, device, loss,
                           batch_size=config.batch_size,
                           epochs=config.epochs,
                           learning_rate=config.learning_rate,
                           weight_decay=config.weight_decay,
                           # plateau_patience=config.plateau_patience,
                           scheduler_step=config.scheduler_step,
                           scheduler_gamma=config.scheduler_gamma,
                           delta=config.delta,
                           plateau_terminate=config.plateau_terminate,
                           print_every=config.print_every,
                           checkpoint_file=checkpoint_file)

    model.load_state_dict(torch.load(checkpoint_file))
    loss_train = eval_wno_1d(model, train_loader, loss, config.batch_size, device)
    loss_val = eval_wno_1d(model, val_loader, loss, config.batch_size, device)
    loss_test = eval_wno_1d(model, test_loader, loss, config.batch_size, device)
    print('loss_train (model saved in checkpoint):', loss_train)
    print('loss_val (model saved in checkpoint):', loss_val)
    print('loss_test (model saved in checkpoint):', loss_test)


@hydra.main(version_base=None, config_path="../config/", config_name="wno")
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg, resolve=True))

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


if __name__ == '__main__':
    main()
