import numpy as np
import gym
import time

np.random.seed(1)

env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]

def choose_action():
    action = np.random.randint(0, N_ACTIONS)
    return action

for i_episode in range(4000):
    s = env.reset()
    ep_r = 0
    for step in range(180):
        env.render()
        a = choose_action()

        # take action
        s_, r, done, info = env.step(a)
        ep_r += r
        s = s_
    print('Ep: ', i_episode,'| Ep_r: ', ep_r)

