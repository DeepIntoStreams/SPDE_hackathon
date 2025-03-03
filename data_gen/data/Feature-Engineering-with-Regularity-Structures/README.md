# Feature-Engineering-with-Regularity-Structures
Classes and Experiments for the paper "Feature Engineering with Regularity Structures".

Authors of the paper: Ilya Chevyrev, Andris Gerasimovics, Hendrik Weber.

Code produced by: Andris Gerasimovics.

Link to the paper: https://arxiv.org/abs/2108.05879

Classes: 

Noise - generates noises and initial conditions for the numerical experimants.

SPDEs - (S)PDE solver of Parabolic, Wave and Burger's equations using finite difference method. Contains integrator functions for the Model class.

Rule - helper class that generates the rule for creating the model feature set.

Model - generates model features.

Algorithm 1 - Regression and Learning as well as a visualization for Algorithm 1.

full_vizualization - visualization class for the Algorithm 2. Allows to compute errors.

Algorithm 2 - full implemintation of the Algorithm 2 in the case of Burger's or Parabolic equation.

Experiments:

Parabolic_Numerics - contains numerical simulation of Algorithm 1 for multiplicative and additive parabolic equations. (Section 3.1 in the paper)

Wave_Numerics - contains numerical simulation of Algorithm 1 for a wave equation. (Section 3.2 in the paper)

Burgers_Numerics - contains numerical simulation of Algorithm 2 for Burger's equation. (Section 3.3 in the paper)
