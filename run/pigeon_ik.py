from env.ik_pigeon_gym import IKPigeon, calc_end_effector_pos, get_jacobian
import numpy as np
import pickle

env = IKPigeon()
for iter_num in range(1):
    x = env.reset()
    for _ in range(100):
        env.render()
        u = env.action_space.sample()
        x, _, done, _ = env.step(u)
        print(x[:2])
        print(calc_end_effector_pos(x[[3, 5, 7]])[-1])
        print(get_jacobian(x[[3, 5, 7]]))
        if done:
            break
env.close()
