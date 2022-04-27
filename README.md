# ik_lqr
Basic classical control theory-related projects.

## Getting Started
Execute the following command in the main directory.
```
export PYTHONPATH=$PWD
```

### Recommended Dependencies
```
python=3.8.12
numpy=1.21.2
tensorflow=2.7.0
matplotlib=3.5.1
control=0.9.1
gym=0.21.0
box2d-py=2.3.8
cma=3.2.2
scipy=1.8.0
```

## Projects
### Continuous Cartpole Control
* Control system modeled using gradient descent on $\dot x = Ax + Bu$
* Cartpole control solver
  * Manual eigenvalue placement
  * Search $T = A - BK$ that maximizes the return using CMA-ES
  * Continuous LQR
* Linear system classification analysis for $T$
* Demo:
```
python run/cartpole.py
```
https://github.com/johnlime/ik_lqr/tree/master/run/continuous_cartpole_dynamics

### Discrete Cartpole Control \[Suboptimal\]
* Control system modeled using gradient descent on $\dot x = Ax + Bu$
* Cartpole control solver
  * Manual eigenvalue placement
  * Search $T = A - BK$ that maximizes the return using CMA-ES
  * Continuous and discrete LQR
* Linear system classification analysis for $T$

https://github.com/johnlime/ik_lqr/tree/master/run/discrete_cartpole_dynamics

### Inverse Kinematics using Pseudoinverse of Jacobian
* Uses the ["pigeon" environment](https://github.com/johnlime/pigeon_head_bob)
* Jacobian matrix calculated [manually](https://github.com/johnlime/ik_lqr/tree/master/doc/main.pdf).
* Demo:
```
python run/pigeon_ik.py
```
