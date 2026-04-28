"""Generate prediction files for evaluation.

Creates files containing {'u_real': Tensor, 'u_pred': Tensor} for each dataset.

Usage:
  python generate_prediction_files.py --model NSPDE1D --checkpoint /path/to.ckpt --data data1.mat data2.mat --out-dir predictions

"""
import os
import os.path as osp
import argparse

import torch
import scipy.io

from evaluations import collect_predictions

# model and dataloader constructors
from model.NSPDE.utilities import dataloader_nspde_1d, dataloader_nspde_2d
from model.FNO.FNO1D import dataloader_fno_1d_xi
from model.FNO.FNO2D import dataloader_fno_2d_xi

# model classes
from model.NSPDE.neural_spde import NeuralSPDE
from model.FNO.FNO1D import FNO_space1D_time
from model.FNO.FNO2D import FNO_space2D_time


CONFIGS = {
    'NSPDE1D': {
        'model_init': lambda: NeuralSPDE(dim=1, in_channels=1, noise_channels=1, hidden_channels=32,
                                         n_iter=1, modes1=64, modes2=50),
        'dataloader': dataloader_nspde_1d,
        'dataloader_args': {'ntrain': 20, 'ntest': 500, 'T': 51, 'sub_t': 1, 'batch_size': 20, 'dim_x': 128},
        'type': 'nspde_1d'
    },
    'NSPDE2D': {
        'model_init': lambda: NeuralSPDE(dim=2, in_channels=1, noise_channels=1, hidden_channels=32,
                                         n_iter=1, modes1=64, modes2=50),
        'dataloader': dataloader_nspde_2d,
        'dataloader_args': {'ntrain': 20, 'ntest': 500, 'T': 51, 'sub_t': 1, 'sub_x': 4, 'batch_size': 20},
        'type': 'nspde_2d'
    },
    'FNO1D': {
        'model_init': lambda: FNO_space1D_time(modes1=64, modes2=50, width=32, L=4, T=51),
        'dataloader': dataloader_fno_1d_xi,
        'dataloader_args': {'ntrain': 20, 'ntest': 500, 'T': 51, 'sub_t': 1, 'batch_size': 20, 'dim_x': 128},
        'type': 'fno_1d'
    },
    'FNO2D': {
        'model_init': lambda: FNO_space2D_time(modes1=64, modes2=50, width=32, L=4, T=51),
        'dataloader': dataloader_fno_2d_xi,
        'dataloader_args': {'ntrain': 20, 'ntest': 500, 'T': 51, 'sub_t': 1, 'sub_x': 4, 'batch_size': 20},
        'type': 'fno_2d'
    },
}


def get_test_loader(model_type, u_data, xi_data):
    cfg = CONFIGS[model_type]
    dl_fn = cfg['dataloader']
    args = cfg['dataloader_args'].copy()
    # adjust args for 2d/3d loaders
    args.pop('dim_x', None)
    _, test_loader = dl_fn(u=u_data, xi=xi_data, **args)
    return test_loader


def make_forward_fn(model_type, model, device):
    t = CONFIGS[model_type]['type']
    if t.startswith('nspde'):
        def forward_fn(batch, device):
            u0, xi, u = [x.to(device) for x in batch]
            return model(u0, xi), u
    elif t.startswith('fno'):
        def forward_fn(batch, device):
            xi, u = [x.to(device) for x in batch]
            out = model(xi)
            # some FNOs return extra channel dim
            if out.ndim == 4 and out.shape[-1] == 1:
                out = out[..., 0]
            return out, u
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return forward_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=list(CONFIGS.keys()))
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', nargs='+', required=True, help='Paths to .mat data files')
    parser.add_argument('--out-dir', default='predictions')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # build model
    model = CONFIGS[args.model]['model_init']().to(args.device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    for data_path in args.data:
        mat = scipy.io.loadmat(data_path)
        W, Sol = mat['W'], mat['sol']
        xi = torch.from_numpy(W.astype('float32'))
        u_data = torch.from_numpy(Sol.astype('float32'))

        test_loader = get_test_loader(args.model, u_data, xi)

        forward_fn = make_forward_fn(args.model, model, args.device)

        # collect predictions (returns CPU tensors)
        u_real, u_pred = collect_predictions(test_loader, forward_fn, torch.device(args.device))

        base = osp.splitext(osp.basename(data_path))[0]
        out_file = osp.join(args.out_dir, f"{args.model}_{base}.pt")
        torch.save({'u_real': u_real, 'u_pred': u_pred}, out_file)
        print(f"Saved predictions to {out_file}")


if __name__ == '__main__':
    main()
