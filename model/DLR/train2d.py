# Adapted from https://github.com/sdogsq/DLR-Net
# Modified for current implementation by the authors of SPDEBench

import time
import hydra
from omegaconf import DictConfig, OmegaConf
import random
import os
import os.path as osp
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(osp.join(current_directory, "..", ".."))
import warnings
warnings.filterwarnings('ignore')

from model.DLR.utils2d import *
from model.utilities import EarlyStopping

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def mytrain(config):

    os.makedirs(config.save_dir, exist_ok=True)
    checkpoint_file = config.save_dir + config.checkpoint_file

    reader = MatReader(config.data_path, to_torch = False)
    data = mat2data(reader, config.sub_t, config.sub_x)
    indices = np.random.permutation(data['Solution'].shape[0])
    print('indices:', indices[:10])
    data['Solution'] = data['Solution'][indices]
    data['W'] = data['W'][indices]

    ntrain, nval, ntest = config.ntrain, config.nval, config.ntest

    _, test_W, _, test_U0, _, test_Y = dataloader_2d(u=data['Solution'], xi=data['W'], ntrain=ntrain + nval,
                                                     ntest=ntest, T=config.T,
                                                     sub_t=config.sub_t, sub_x=config.sub_x)
    train_W, val_W, train_U0, val_U0, train_Y, val_Y = dataloader_2d(u=data['Solution'][:ntrain + nval],
                                                                     xi=data['W'][:ntrain + nval],
                                                                     ntrain=ntrain, ntest=ntest, T=config.T,
                                                                     sub_t=config.sub_t, sub_x=config.sub_x)
    print(f"train_W: {train_W.shape}, train_U0: {train_U0.shape}, train_Y: {train_Y.shape}")
    print(f"val_W: {val_W.shape}, val_U0: {val_U0.shape}, val_Y: {val_Y.shape}")
    print(f"test_W: {test_W.shape}, test_U0: {test_U0.shape}, test_Y: {test_Y.shape}")
    print(f"data['T']: {data['T'].shape}, data['X']: {data['X'].shape}, data['Y']: {data['Y'].shape}")

    graph = NS_graph(data, config.height)
    for key, item in graph.items():
        print(key, item)
    print("Total Feature Number:", len(graph))

    train_W, train_U0, train_Y = torch.Tensor(train_W), torch.Tensor(train_U0), torch.Tensor(train_Y)
    val_W, val_U0, val_Y = torch.Tensor(val_W), torch.Tensor(val_U0), torch.Tensor(val_Y)
    test_W, test_U0, test_Y = torch.Tensor(test_W), torch.Tensor(test_U0), torch.Tensor(test_Y)

    model = rsnet_2d(graph, data['T'], X=data['X'][:, 0], Y=data['Y'][0], nu=config.nu).to(device)
    print("Trainable parameter number: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # cache Xi fatures
    Feature_Xi = cacheXiFeature_2d(graph, T=data['T'], X=data['X'][:, 0], Y=data['Y'][0],
                                   W=train_W, eps=config.nu, device=device)
    trainset = TensorDataset(train_W, train_U0, Feature_Xi, train_Y)
    train_loader = DataLoader(trainset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              persistent_workers=True,
                              drop_last=True,
                              num_workers=4)

    val_F_Xi = cacheXiFeature_2d(graph, T=data['T'], X=data['X'][:, 0], Y=data['Y'][0],
                                  W=val_W, eps=config.nu, device=device)
    valset = TensorDataset(val_W, val_U0, val_F_Xi, val_Y)
    val_loader = DataLoader(valset,
                             batch_size=config.batch_size,
                             shuffle=False,
                             pin_memory=True,
                             persistent_workers=True,
                             drop_last=False,
                             num_workers=4)

    test_F_Xi = cacheXiFeature_2d(graph, T=data['T'], X=data['X'][:, 0], Y=data['Y'][0],
                                  W=test_W, eps=config.nu, device=device)

    testset = TensorDataset(test_W, test_U0, test_F_Xi, test_Y)
    test_loader = DataLoader(testset,
                             batch_size=config.batch_size,
                             shuffle=False,
                             pin_memory=True,
                             persistent_workers=True,
                             drop_last=False,
                             num_workers=4)

    lossfn = LpLoss(size_average=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs, verbose = False)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    trainTime = 0
    early_stopping = EarlyStopping(patience=config.plateau_terminate,
                                   verbose=False,
                                   delta=config.delta,
                                   path=config.checkpoint_file)
    for epoch in range(1, config.epochs + 1):

        # ------ train ------
        tik = time.time()
        trainLoss = train(model, device, train_loader, optimizer, lossfn, epoch)
        tok = time.time()
        trainTime += tok - tik

        scheduler.step()

        valLoss = test(model, device, val_loader, lossfn, epoch)

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

    testLoss = test(model, device, test_loader, lossfn)

    print('loss_test:', testLoss)


@hydra.main(version_base=None, config_path="../config/", config_name="dlr_ns")
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
