# Adapted from https://github.com/sdogsq/DLR-Net

import time
import torch
import torch.optim as optim
import wandb
import scipy.io
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from timeit import default_timer
import random
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import os
import os.path as osp
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(osp.join(current_directory, "..", ".."))
import warnings
warnings.filterwarnings('ignore')

from model.DLR.utils2d import *
from model.utilities import EarlyStopping, plot_2d_xi

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def mytrain(cfg):
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        config={**cfg.fixed.data,
                **cfg.fixed.model,
                **cfg.fixed.train,
                **cfg.fixed.save},
        # mode="disabled"
    )

    os.makedirs(wandb.config.base_dir, exist_ok=True)
    cp = f"{wandb.config.checkpoint_file}_run-{wandb.run.id}.pth"
    run.config.update({'checkpoint_file': f'{wandb.config.base_dir}/{cp}'}, allow_val_change=True)

    print("Current WandB Config:", dict(wandb.config))

    # Set random seed
    seed = wandb.config.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # data = scipy.io.loadmat(wandb.config.data_path)
    reader = MatReader(wandb.config.data_path, to_torch = False)
    data = mat2data(reader, wandb.config.sub_t, wandb.config.sub_x)
    indices = np.random.permutation(data['Solution'].shape[0])
    print('indices:', indices[:10])
    data['Solution'] = data['Solution'][indices]
    data['W'] = data['W'][indices]

    ntrain = wandb.config.ntrain
    # nval = wandb.config.nval
    ntest = wandb.config.ntest

    train_W, test_W, train_U0, test_U0, train_Y, test_Y = dataloader_2d(
                                                            u=data['Solution'], xi=data['W'], ntrain=ntrain,
                                                            ntest = ntest, T = wandb.config.T,
                                                            sub_t = wandb.config.sub_t, sub_x = wandb.config.sub_x)
    print(f"train_W: {train_W.shape}, train_U0: {train_U0.shape}, train_Y: {train_Y.shape}")
    print(f"test_W: {test_W.shape}, test_U0: {test_U0.shape}, test_Y: {test_Y.shape}")
    print(f"data['T']: {data['T'].shape}, data['X']: {data['X'].shape}, data['Y']: {data['Y'].shape}")

    graph = NS_graph(data, wandb.config.height)
    for key, item in graph.items():
        print(key, item)
    print("Total Feature Number:", len(graph))

    train_W, train_U0, train_Y = torch.Tensor(train_W), torch.Tensor(train_U0), torch.Tensor(train_Y)
    test_W, test_U0, test_Y = torch.Tensor(test_W), torch.Tensor(test_U0), torch.Tensor(test_Y)

    model = rsnet_2d(graph, data['T'], X=data['X'][:, 0], Y=data['Y'][0], nu=wandb.config.nu).to(device)
    print("Trainable parameter number: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    if cfg.load_cp:
        model.load_state_dict(torch.load(cfg.load_cp))
        print('Load checkpoint from:', cfg.load_cp)

    # cache Xi fatures
    Feature_Xi = cacheXiFeature_2d(graph, T=data['T'], X=data['X'][:, 0], Y=data['Y'][0],
                                   W=train_W, eps=wandb.config.nu, device=device)
    print(Feature_Xi.shape)
    trainset = TensorDataset(train_W, train_U0, Feature_Xi, train_Y)
    train_loader = DataLoader(trainset,
                              batch_size=wandb.config.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              persistent_workers=True,
                              drop_last=True,
                              num_workers=4)

    test_F_Xi = cacheXiFeature_2d(graph, T=data['T'], X=data['X'][:, 0], Y=data['Y'][0],
                                  W=test_W, eps=wandb.config.nu, device=device)

    testset = TensorDataset(test_W, test_U0, test_F_Xi, test_Y)
    test_loader = DataLoader(testset,
                             batch_size=wandb.config.batch_size,
                             shuffle=False,
                             pin_memory=True,
                             persistent_workers=True,
                             drop_last=False,
                             num_workers=4)

    lossfn = LpLoss(size_average=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)
    if wandb.config.sche == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, wandb.config.epochs, verbose = False)
    elif wandb.config.sche == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    trainTime = 0
    # early_stopping = EarlyStopping(patience=wandb.config.plateau_terminate,
    #                                verbose=False,
    #                                delta=wandb.config.delta,
    #                                path=wandb.config.checkpoint_file)
    for epoch in range(1, wandb.config.epochs + 1):

        # ------ (1) train ------
        tik = time.time()
        trainLoss = train(model, device, train_loader, optimizer, lossfn, epoch)
        tok = time.time()
        trainTime += tok - tik

        scheduler.step()

        testLoss = test(model, device, test_loader, lossfn, epoch)

        # early_stopping(testLoss, model)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

        run.log({'Train Loss': trainLoss, 'Test Loss': testLoss})

        if (epoch-1) % wandb.config.print_every == 0:
            print('Epoch: {:04d} \tTrain Loss: {:.6f} \tTest Loss: {:.6f} \t\
                               Average Training Time per Epoch: {:.3f} \t' \
                  .format(epoch, trainLoss, testLoss, trainTime / epoch))

    ## ----------- test ------------
    # model.load_state_dict(torch.load(wandb.config.checkpoint_file))

    torch.save(model.state_dict(), wandb.config.checkpoint_file)

    run.summary['loss_test'] = testLoss

    # plot_2d_xi(model, test_loader, device, T=wandb.config.T // wandb.config.sub_t, wb=run)

    run.finish()


@hydra.main(version_base=None, config_path="../config/", config_name="dlr_ns")
def main(cfg: DictConfig):
    mytrain(cfg)


if __name__ == '__main__':
    main()
