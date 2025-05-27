# SPDEBench: An Extensive Benchmark for Learning Regular and Singular Stochastic PDEs

This repository is the official implementation
of [SPDEBench: An Extensive Benchmark for Learning Regular and Singular Stochastic PDEs](https://github.com/DeepIntoStreams/SPDE_hackathon).

SPDEBench is designed to solve typical SPDEs of physical significance (i.e.
the $\Phi^4_d$, wave, incompressible Navier-Stokes, and KdV equations) on 1D or 2D tori driven by white noise via ML
methods. New datasets for singular SPDEs based on the renormalization process have been constructed, and novel ML models
achieving the best results to date have been proposed. Results are benchmarked with traditional numerical solvers and
ML-based models, including FNO, NSPDE and DLR-Net, etc. 

Below, we provide instructions on how to use code in this repository to generate datasets and train models as in our paper.

![Phi42](https://github.com/DeepIntoStreams/SPDE_hackathon/blob/main/Phi42_xi_eps_128_sigma_1_249.png)

---

## Requirements

The code has been tested with Python 3.8 and PyTorch 2.4.1 (CUDA 11.8). For Linux users, we recommend installing PyTorch with the following command:
```setup
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia
```
Alternatively, you can refer to the [official PyTorch website](https://pytorch.org/get-started/previous-versions/) for other installation options.

To install the remaining dependencies:
```setup
pip install -r requirements.txt
```

---

## Data Generation

To generate the data, run the corresponding python files in `data_gen/examples/`. For instance, to generate data
of $\Phi^4_2$ equation with varying initial conditions and noise truncation degree 128, run the following:

```bash
python gen_phi42.py sim.fix_u0=False sim.eps=128
```

Settings of the equation can be tailored by choose different values for config args.

- `a`,`b`,`Nx` (and `c`,`d`,`Ny` in 2D case): begin point, end point, space resolution.
- `s`,`t`,`Nt`: start time, end time, time resolution.
- `truncation` or `eps`: truncation degree of noise.
- `sigma`: coefficient in the additive diffusive term.
- `fix_u0`: `True`--fix initial condition; `False`--vary initial condition among samples.
- `num`: number of samples generated.
- `sub_x`,`sub_t`
- `save_dir`
- `save_name`

(More details about the config args will be added later.)

---

## Models

This repository contains code for seven ML models: NCDE, NRDE, NCDE-FNO, DeepONet, FNO, NSPDE, and a novel ML model
called NSPDE-S.
To train the model, run `train1d.py` or `train2d.py` in the corresponding folder named by the model.
For instance, after setting proper config args in corresponding `.yaml` file, run the following:

```bash
python train1d.py
```

### Brief introduction to key config args in models

- `task`: `xi` (or `u0xi` if applicable)
- `data_path`: Directory where the datasets are saved.
- `dim_x`: Number of points in space dimension.
- `T`: Total number of time steps (i.e. time sequence length).
- `sub_t`: Subsampling interval. Use all time steps in data if sub_t=1, or sample every sub_t steps to reduce data density.
- `ntrain`,`nval`,`ntest`: Number of samples in the training, validation, and test sets.
- `epochs`: Total number of training epochs.
- `batch_size`: Number of samples per batch.
- `learning_rate`: Initial learning rate.
- `weight_decay`: Used in the optimizer.
- `scheduler_step`: Interval (in epochs) for learning rate adjustment.
- `scheduler_gamma`: Learning rate decay factor. At each adjustment, the learning rate is multiplied by this value.
- `plateau_patience`: Used in some scheduler to control the reduction of learning rate.
- `plateau_terminate`: Early stop the training if validation loss doesn't improve after such number of epochs.
- `delta`: Minimum threshold for improvement.
- `print_every`: Training log frequency.
- `save_dir`: Directory where output files (i.e. checkpoints) will be saved.
- `checkpoint_file`: File name of model checkpoints (.pth).
- `log_file`: File where results of hyperparameter search will be logged.

(More details about the config args will be added later.)

#### NCDE specific args:

- `hidden_channels`
- `solver`

#### NRDE specific args:

- `hidden_channels`
- `solver`
- `depth`
- `window_length`

#### NCDE-FNO specific args:

- `hidden_channels`
- `solver`

#### DeepONet specific args:

- `width`
- `branch_depth`
- `trunk_depth`

#### FNO specific args:

- `L`
- `modes1`
- `modes2`

#### NSPDE / NSPDE-S specific args:

- `hidden_channels`
- `n_iter`
- `modes1`
- `modes2`

---

## Directory Structure

```
SPDE_hackathon          
├───data_gen
│   ├───configs    # YAML configuration files specifying parameters for data generation (.yaml)
│   │       
│   ├───examples
│   │       gen_KdV.py
│   │       gen_navier_stokes.py
│   │       gen_phi41.py
│   │       gen_phi42.py
│   │       gen_wave.py
│   │                 
│   ├───notebook    # Jupyter notebooks for visualizing the generated data (.ipynb)
│   │       
│   └───src    # Core scripts for SPDE solver generation (.py).
│           
└───model
    │   utilities.py
    │   
    ├───config    # All the config files for models
    │       
    ├───DeepONet
    │       deepOnet.py
    │       train1d.py
    │       
    ├───DLR
    │       Graph.py
    │       RSlayer.py
    │       RSlayer_2d.py
    │       Rule.py
    │       SPDEs.py
    │       train1d.py
    │       train2d.py
    │       utils.py
    │       utils2d.py
    │       
    ├───FNO
    │       FNO1D.py
    │       FNO2D.py
    │       train1d.py
    │       train2d.py
    │       
    ├───NCDE
    │       NCDE.py
    │       train1d.py
    │       
    ├───NCDEFNO
    │       NCDEFNO_1D.py
    │       train1d.py
    │       
    ├───NRDE
    │       NRDE.py
    │       train1d.py
    │       
    └───NSPDE
            diffeq_solver.py
            fixed_point_solver.py
            gradients.py
            linear_interpolation.py
            neural_aeps_spde.py
            neural_spde.py
            Noise.py
            root_find_solver.py
            root_finding_algorithms.py
            SPDEs2D.py
            train1d.py
            train2d.py
            train2d_aeps.py
            train2d_alleps.py
            utilities.py
            utilities_aeps.py
```

---

## Acknowledgements 
This project incorporates code from the following open-source repositories:

- [Feature Engineering with Regularity Structures](https://github.com/andrisger/Feature-Engineering-with-Regularity-Structures).
- [NeuralCDE](https://github.com/patrick-kidger/NeuralCDE). Licensed under the Apache-2.0 license.
- [Fourier Neural Operator](https://github.com/li-Pingan/fourier-neural-operator). Licensed under the MIT license.
- [torchcde](https://github.com/patrick-kidger/torchcde.git). Licensed under the Apache-2.0 license.
- [DEQ](https://github.com/locuslab/deq.git). Licensed under the MIT license.
- [Neural-SPDEs](https://github.com/crispitagorico/torchspde). Licensed under the Apache-2.0 license.
- [DLR-Net](https://github.com/sdogsq/DLR-Net).

Many thanks to their authors for sharing these valuable contributions!