# Examples

## Data generation for the stochastic Ginzburg-Landau equation

The Stochastic Ginzburg-Landau equation is also known as the Allen-Cahn equation in 1-dimension and is used for modeling various physical phenomena like superconductivity.

In `gen_ginzburg_landau.py` we generate solutions of the stochastic Ginzburg-Ladau equations,

$$
\partial_t u - \Delta u = 3u -u^3 + \xi, \quad
    u(t,0) = u(t,1), \quad
    u(0,x) = u_0(x), \quad 
    (t,x)\in [0,T] \times [0,1].
$$

with the initial condition $u_0$ either being fixed across samples ($u_0=x(1-x)$, for example), or varying.

We sample multiple paths $\xi^1, \ldots, \xi^n$ from a cylindrical Wiener process in one dimension, and then solve the SPDE (using the finite difference method).

## Data generation for the stochastic Korteweg–De Vries (KdV) equation

In `gen_KdV.py` we generate solutions of the stochastic Korteweg–De Vries (KdV) equations,

$$
\partial_t u + \gamma \partial_x^3 u = 6u \partial_x u + \xi,\quad
    u(t,0) = u(t,1), \quad
    u(0,x) = u_0(x), \quad 
    (t,x)\in [0,T] \times [0,1].
$$

with the initial condition $u_0$ either being fixed across samples ($u_0=x(1-x)$, for example), or varying.

We sample multiple paths $\xi^1, \ldots, \xi^n$ from a Q-Wiener process in one dimension, and then solve the SPDE (using the spectral Galerkin method).
