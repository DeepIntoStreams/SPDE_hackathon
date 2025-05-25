# Adapted from https://github.com/sdogsq/DLR-Net
# Modified for current implementation by the authors of SPDEBench

import time
import scipy.io
import hydra
from omegaconf import DictConfig, OmegaConf
import random
from sklearn.model_selection import train_test_split
import os
import os.path as osp
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(osp.join(current_directory, "..", ".."))
import warnings
warnings.filterwarnings('ignore')

from model.DLR.utils import *
from model.utilities import EarlyStopping

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def mytrain(config):

    os.makedirs(config.save_dir, exist_ok=True)
    checkpoint_file = config.save_dir + config.checkpoint_file

    data = scipy.io.loadmat(config.data_path)
    data['T'] = data['T'].squeeze()
    data['X'] = data['X'].squeeze()
    data['W'] = data['W'].transpose(0,2,1)
    data['Soln_add'] = data['sol'].transpose(0,2,1)
    data['U0'] = data['Soln_add'][:,0,:]

    ntrain, nval, ntest = config.ntrain, config.nval, config.ntest

    train_W, test_W, train_U0, test_U0, train_Y, test_Y = train_test_split(data['W'],  # noise: [N,time,space]
                                                                           data['U0'],  # initial sol: [N,space]
                                                                           data['Soln_add'],  # sol: [N,time,space]
                                                                           train_size=ntrain,
                                                                           test_size = nval + ntest,
                                                                           shuffle=False)
    val_W, test_W, val_U0, test_U0, val_Y, test_Y = train_test_split(test_W,
                                                                     test_U0,
                                                                     test_Y,
                                                                     train_size=nval,
                                                                     shuffle=False)
    print(f"train_W: {train_W.shape}, train_U0: {train_U0.shape}, train_Y: {train_Y.shape}")
    print(f"val_W: {val_W.shape}, val_U0: {val_U0.shape}, val_Y: {val_Y.shape}")
    print(f"test_W: {test_W.shape}, test_U0: {test_U0.shape}, test_Y: {test_Y.shape}")

    graph = parabolic_graph(data, height=config.height)
    print(graph)

    train_W, train_U0, train_Y = torch.Tensor(train_W), torch.Tensor(train_U0), torch.Tensor(train_Y)
    val_W, val_U0, val_Y = torch.Tensor(val_W), torch.Tensor(val_U0), torch.Tensor(val_Y)
    test_W, test_U0, test_Y = torch.Tensor(test_W), torch.Tensor(test_U0), torch.Tensor(test_Y)

    # cache Xi fatures
    Feature_Xi = cacheXiFeature(graph, T = data['T'], X = data['X'], W = train_W, device = device)

    trainset = TensorDataset(train_W, train_U0, Feature_Xi, train_Y)
    train_loader = DataLoader(trainset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              persistent_workers=True,
                              drop_last=True,
                              num_workers=4)

    val_F_Xi = cacheXiFeature(graph, T = data['T'], X = data['X'], W = val_W, device = device)

    valset = TensorDataset(val_W, val_U0, val_F_Xi, val_Y)
    val_loader = DataLoader(valset,
                             batch_size=100,
                             shuffle=True,
                             pin_memory=True,
                             persistent_workers=True,
                             drop_last=False,
                             num_workers=4)

    model = rsnet(graph, data['T'], data['X']).to(device)
    lossfn = LpLoss(size_average=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs, verbose = False)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    trainTime = 0
    early_stopping = EarlyStopping(patience=config.plateau_terminate,
                                   verbose=False,
                                   delta=config.delta,
                                   path=checkpoint_file)
    for epoch in range(1, config.epochs + 1):

        # ------ train ------
        tik = time.time()
        trainLoss = train(model, device, train_loader, optimizer, lossfn, epoch)
        tok = time.time()
        trainTime += tok - tik

        scheduler.step()

        valLoss = test(model, device, val_loader, lossfn)

        early_stopping(valLoss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        if (epoch-1) % config.print_every == 0:
            print('Epoch: {:04d} \tTrain Loss: {:.6f} \tVal Loss: {:.6f} \t\
                               Average Training Time per Epoch: {:.3f} \t' \
                  .format(epoch, trainLoss, valLoss, trainTime / epoch))

    ## ----------- test ------------
    model.load_state_dict(torch.load(checkpoint_file))

    test_F_Xi = cacheXiFeature(graph, T=data['T'], X=data['X'], W=test_W, device=device)

    testset = TensorDataset(test_W, test_U0, test_F_Xi, test_Y)
    test_loader = DataLoader(testset,
                             batch_size=100,
                             shuffle=True,
                             pin_memory=True,
                             persistent_workers=True,
                             drop_last=False,
                             num_workers=4)

    testLoss = test(model, device, test_loader, lossfn)
    print(f'Final Test Loss: {testLoss:.6f}')


@hydra.main(version_base=None, config_path="../config/", config_name="dlr_phi41")
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

    mytrain(cfg)


if __name__ == '__main__':
    main()
