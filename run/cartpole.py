import gym
import numpy as np
import pickle

with open('run/cartpole_system_model/best_controller.pkl', 'rb') as filepath:
    controller_dict = pickle.load(filepath)
K = controller_dict['K']

env = gym.make('CartPole-v1')
x = env.reset()
for _ in range(1000):
    env.render()
    u = np.matmul(-K, x)
    if u < 0:
        u = 0
    else:
        u = 1
    x, _, done, _ = env.step(u)
env.close()
