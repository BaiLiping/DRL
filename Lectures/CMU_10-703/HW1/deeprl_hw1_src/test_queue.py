#!/usr/bin/env python
# coding: utf-8
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from builtins import input

import deeprl_hw1.queue_envs
import gym
import time
import numpy as np
# import random

def run_random_policy(env):
    """Run a random policy for the given environment.

    Logs the total reward and the number of steps until the terminal
    state was reached.

    Parameters
    ----------
    env: gym.envs.Environment
      Instance of an OpenAI gym.

    Returns
    -------
    (float, int)
      First number is the total undiscounted reward received. The
      second number is the total number of actions taken before the
      episode finished.
    """
    initial_state = env.reset()
    # env.render()
    # time.sleep(1)  # just pauses so you can see the output

    total_reward = 0
    num_steps = 0
    while True:
        action_cur = env.action_space.sample()
        nextstate, reward, is_terminal, debug_info = env.step(action_cur)
        env.render()

        total_reward += reward
        num_steps += 1

        if is_terminal:
            break

        time.sleep(5)

    return total_reward, num_steps


def main():
    # create the environment
    env = gym.make('Queue-1-v0')
    # env = gym.make('Queue-2-v0')
    # initialization
    gamma = 0.9
    # tol = 1e-3
    # max_iterations = 1000
    # grid_width = 4
    # env.render()

    ##############################################################################################    
    # # run an episode test using random policy 
    input('Hit enter to run an iteration based on random policy...')
    total_reward, num_steps = run_random_policy(env)
    print('Agent received total cumulative discounted reward of: %f' % (total_reward * np.power(gamma, num_steps - 1)))
    print('Agent took %d steps' % num_steps)


if __name__ == '__main__':
    main()


