import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    x = env.render()
    u = env.action_space.sample() # take a random action
    x_prime, _, done, _ = env.step(u)
env.close()
