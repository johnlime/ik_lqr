# ik_lqr
Basic classical control theory-related projects.

## Cartpole control
* Control system modeled using gradient descent on $\dot x = Ax + Bu$
* Cartpole control solver
  * Manual eigenvalue placement
  * Search $T = A - BK$ that maximizes the return using CMA-ES
  * Continuous and discrete LQR
* Linear system classification analysis for $T$

https://github.com/johnlime/ik_lqr/tree/master/run/cartpole_dynamics
