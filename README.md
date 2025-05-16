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

To install requirements:

```setup
pip install -r requirements.txt
```

---

## Data Generation

To generate the data, run the corresponding python files in `data_gen/examples/`. For instance, to generate data
of $\Phi^4_2$ equation with varying initial conditions and noise truncation degree 128, run the following:

```bash
python gen_phi42.py fix_u0=False eps=128
```

Settings of the equation can be tailored by choose different values for config args.

- `fix_u0`: `True`--fix initial condition; `False`--vary initial condition among samples.
- `a`,`b`,`Nx` (and `c`,`d`,`Ny` in 2D case): begin point, end point, space resolution.
- `s`,`t`,`Nt`: start time, end time, time resolution.
- `truncation`: truncation degree of noise.
- `sigma`:
- `num`: number of samples generated.

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
- `dim_x`: Dimension of the space variable.
- `T`: Total number of time steps (i.e. time sequence length).
- `sub_t`: Subsampling interval. Use all time steps in data if sub_t=1, or sample every sub_t steps to reduce data density.
- `ntrain`,`nval`,`ntest`: Number of samples in the training, validation, and test sets.
- `num_workers`: 
- `epochs`: Total number of training epochs.
- `batch_size`: Number of samples per batch.
- `learning_rate`: Initial learning rate.
- `scheduler_step`: Interval (in epochs) for learning rate adjustment.
- `scheduler_gamma`: Learning rate decay factor. At each adjustment, the learning rate is multiplied by this value.
- `plateau_patience`: 
- `plateau_patience`
- `delta`: Minimum threshold forimprovement.
- `print_every`: Training log frequency.
- `base_dir`: Directory where output files (i.e. checkpoints) will be saved.
- `checkpoint_file`: File name of model checkpoints (.pth).

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
    │       deepOnet2D.py
    │       train1d.py
    │       train2d.py
    │       
    ├───DLR
    │       Graph.py
    │       RSlayer.py
    │       RSlayer_2d.py
    │       Rule.py
    │       SPDEs.py
    │       train1d_phi41.py
    │       train2d_NS.py
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
    │       NCDEFNO_2D.py
    │       train1d.py
    │       
    ├───NRDE
    │       NRDE.py
    │       train.py
    │       train_sweep.py
    │       train_wb.py
    │       utils.py
    │       
    └───NSPDE
            diffeq_solver.py
            fixed_point_solver.py
            gradients.py
            linear_interpolation.py
            neural_spde.py
            neural_aeps_spde.py
            Noise.py
            SPDEs2D.py
            root_finding_algorithms.py
            root_find_solver.py
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

[Feature Engineering with Regularity Structures](https://github.com/andrisger/Feature-Engineering-with-Regularity-Structures)

[Fourier Neural Operator](https://github.com/li-Pingan/fourier-neural-operator)

[Neural-SPDEs](https://github.com/crispitagorico/torchspde)

[DLR-Net](https://github.com/sdogsq/DLR-Net)

Many thanks to their authors for sharing these valuable contributions!