from env.ik_pigeon_gym import IKPigeon
import numpy as np
import pickle

env = IKPigeon()
for iter_num in range(1):
    x = env.reset()
    print(x) # 45 deg is 0
    for _ in range(1000):
        env.render()
        u = env.action_space.sample()
        x, _, done, _ = env.step(u)
        print(x)
        if done:
            break
env.close()
