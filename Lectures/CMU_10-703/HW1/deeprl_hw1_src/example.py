#!/usr/bin/env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from builtins import input

import deeprl_hw1.lake_envs as lake_env
import gym
import time as time
import numpy as np
import random
from deeprl_hw1.rl import policy_iteration 
from deeprl_hw1.rl import evaluate_policy 
from deeprl_hw1.rl import print_value_function
from deeprl_hw1.rl import print_policy
from deeprl_hw1.rl import value_function_to_policy
from deeprl_hw1.rl import improve_policy
from deeprl_hw1.rl import value_iteration

action_names = {0: 'LEFT', 2: 'RIGHT', 1: 'DOWN', 3: 'UP'} 

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
    env.render()
    time.sleep(1)  # just pauses so you can see the output

    total_reward = 0
    num_steps = 0
    while True:
        action_cur = env.action_space.sample()
        print(" ")
        print("action is %s" % action_names[action_cur])
        nextstate, reward, is_terminal, debug_info = env.step(action_cur)

        env.render()
        print("move to state %d" % nextstate) 

        total_reward += reward
        num_steps += 1

        if is_terminal:
            break

        time.sleep(3)

    return total_reward, num_steps

def run_policy_iteration(env):
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
    env.render()
    time.sleep(1)  # just pauses so you can see the output

    total_reward = 0
    num_steps = 0
    gamma = 0.9
    tol = 1e-3
    max_iterations = 1000
    state = initial_state

    policy, value_func, iteration_improvement, iteration_evaluation = policy_iteration(env, gamma, max_iterations, tol)
    while True:
        action_cur = policy[state]
        print(" ")
        print("step %d" % num_steps)
        print("action is %s" % action_names[action_cur])
        nextstate, reward, is_terminal, debug_info = env.step(action_cur)
        print(debug_info)
        state = nextstate
        env.render()
        print("move to state %d" % nextstate) 

        total_reward += reward
        num_steps += 1

        if is_terminal:
            break

        time.sleep(1)

    return total_reward, num_steps

def run_value_iteration(env):
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
    env.render()
    # time.sleep(1)  # just pauses so you can see the output

    total_reward = 0
    num_steps = 0
    gamma = 0.9
    tol = 1e-3
    max_iterations = 1000
    state = initial_state

    optimal_value_function, iterations = value_iteration(env, gamma, max_iterations, tol)
    policy = value_function_to_policy(env, gamma, optimal_value_function)

    while True:
        action_cur = policy[state]
        print(" ")
        print("step %d" % num_steps)
        print("action is %s" % action_names[action_cur])
        nextstate, reward, is_terminal, debug_info = env.step(action_cur)
        print(debug_info)
        state = nextstate
        env.render()
        print("move to state %d" % nextstate) 

        total_reward += reward
        num_steps += 1

        if is_terminal:
            break

        # time.sleep(1)

    return total_reward, num_steps


def print_env_info(env):
    print('Environment has %d states and %d actions.' % (env.nS, env.nA))


def print_model_info(env, state, action):
    transition_table_row = env.P[state][action]
    print(
        ('According to transition function, '
         'taking action %s(%d) in state %d leads to'
         ' %d possible outcomes') % (lake_env.action_names[action],
                                     action, state, len(transition_table_row)))
    for prob, nextstate, reward, is_terminal in transition_table_row:
        state_type = 'terminal' if is_terminal else 'non-terminal'
        print(
            '\tTransitioning to %s state %d with probability %f and reward %f'
            % (state_type, nextstate, prob, reward))


def main():
    # create the environment
    env = gym.make('FrozenLake-v0')
    # uncomment next line to try the deterministic version
    env = gym.make('Deterministic-4x4-neg-reward-FrozenLake-v0')
    # env = gym.make('Deterministic-4x4-FrozenLake-v0')
    # env = gym.make('Deterministic-8x8-FrozenLake-v0')
    # env = gym.make('Stochastic-4x4-FrozenLake-v0')
    # env = gym.make('Stochastic-8x8-FrozenLake-v0')



    # print_env_info(env)
    # print_model_info(env, 0, lake_env.DOWN)
    # print_model_info(env, 1, lake_env.DOWN)
    # print_model_info(env, 14, lake_env.RIGHT)

    # initialization
    gamma = 0.9
    tol = 1e-3
    max_iterations = 1000
    grid_width = 4
    env.render()

    # UNIT TEST

    ##############################################################################################
    # test random policy
    # input('Hit enter to run a random policy...')
    # total_reward, num_steps = run_random_policy(env)
    # print('Agent received total reward of: %f' % total_reward)
    # print('Agent took %d steps' % num_steps)


    ##############################################################################################
    # test policy evaluation (deterministic)    
    # policy = np.zeros(env.nS, dtype = np.int)
    # for i in xrange(env.nS):
    #     policy[i] = random.randint(0, 3)
    # print("initial policy is:")
    # print_policy(policy, action_names, grid_width)
    # value_function, num_iterations = evaluate_policy(env, gamma, policy, max_iterations, tol)
    # print("number of iteration needed is %d" % num_iterations)
    # print("value function based on current policy is:")
    # print_value_function(value_function, grid_width)     # 4x4


    ##############################################################################################
    # test policy improvement
    # policy = np.zeros(env.nS, dtype = np.int)
    # print("initial policy is:")
    # print_policy(policy, action_names, grid_width)
    # value_function = np.array([0,     0,    0,     0, 
    #                            0,     0,    0,     0,
    #                            0.729, 0.81, 0.729, 0,
    #                            0.0,   0.9,  1.0,   0])
    # print("value function is:")
    # print_value_function(value_function, grid_width)
    # improved, new_policy = improve_policy(env, gamma, value_function, policy)
    # print("new policy is:")
    # print_policy(new_policy, action_names, grid_width)
    # print("Is policy improved ? %s" % improved)


    ##############################################################################################
    # test policy iteration
    # start_time = time.time()
    # policy, value_func, iteration_improvement, iteration_evaluation = policy_iteration(env, gamma, max_iterations, tol)
    # end_time = time.time()
    # print("final value function is:")
    # print_value_function(value_func, grid_width)
    # print("optimal policy is:")
    # print_policy(policy, action_names, grid_width)
    # print("execution time is %s second" % (end_time - start_time))
    # print("total number of policy improvement is: %d" % iteration_improvement)
    # print("total number of policy evaluation iteration is: %d" % iteration_evaluation)


    ##############################################################################################    
    # test value iteration
    start_time = time.time()
    optimal_value_function, iteration = value_iteration(env, gamma, max_iterations, tol)
    end_time = time.time()
    policy = value_function_to_policy(env, gamma, optimal_value_function)
    print("final value function is:")
    print_value_function(optimal_value_function, grid_width)
    print("optimal policy is:")
    print_policy(policy, action_names, grid_width)
    print("execution time is %s second" % (end_time - start_time))
    print("total number of iteration is: %d" % iteration)


    ##############################################################################################    
    # # run an episode test using optimal policy learned from policy iteration
    # input('Hit enter to run a policy iteration...')
    # total_reward, num_steps = run_policy_iteration(env)
    # print('Agent received total cumulative discounted reward of: %f' % (total_reward * np.power(gamma, num_steps - 1)))
    # print('Agent took %d steps' % num_steps)


    ##############################################################################################    
    # run numbers of episode test using optimal policy learned from policy iteration
    # discounted = 0
    # num_episode = 1000
    # for i in xrange(num_episode):
    #     total_reward, num_steps = run_value_iteration(env)
    #     discounted += total_reward * np.power(gamma, num_steps - 1)  
    # print('Agent received total cumulative discounted reward of: %f' % (discounted / num_episode))



if __name__ == '__main__':
    main()


# state representation
#   0   1   2   3
#   4   5   6   7
#   8   9   10  11
#   12  13  14  15

# action representation
#
#       3
#   0       2
#       1
