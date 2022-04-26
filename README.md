# ik_lqr
Basic classical control theory-related projects.

## Continuous Cartpole Control
* Control system modeled using gradient descent on $\dot x = Ax + Bu$
* Cartpole control solver
  * Manual eigenvalue placement
  * Search $T = A - BK$ that maximizes the return using CMA-ES
  * Continuous LQR
* Linear system classification analysis for $T$

https://github.com/johnlime/ik_lqr/tree/master/run/continuous_cartpole_dynamics

## Discrete Cartpole Control \[Suboptimal\]
* Control system modeled using gradient descent on $\dot x = Ax + Bu$
* Cartpole control solver
  * Manual eigenvalue placement
  * Search $T = A - BK$ that maximizes the return using CMA-ES
  * Continuous and discrete LQR
* Linear system classification analysis for $T$

https://github.com/johnlime/ik_lqr/tree/master/run/discrete_cartpole_dynamics