from env.pigeon_gym import PigeonEnv3Joints
import numpy as np
import pickle

env = PigeonEnv3Joints()
for iter_num in range(1):
    x = env.reset()
    print(x) # 45 deg is 0
    for _ in range(1):
        env.render()
        u = env.action_space.sample()
        x, _, done, _ = env.step(u)
        if done:
            break
env.close()
