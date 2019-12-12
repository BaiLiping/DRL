# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np

# TODO: check the extra import after finished
import time
# from gym.envs.toy_text.frozen_lake import LEFT, RIGHT, DOWN, UP
# action_names = {LEFT: 'LEFT', RIGHT: 'RIGHT', DOWN: 'DOWN', UP: 'UP'} 
action_names = {0: 'LEFT', 2: 'RIGHT', 1: 'DOWN', 3: 'UP'} 

def evaluate_policy(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Evaluate the value of a policy.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """


    total_iteration = 0
    value_function = np.zeros(env.nS, dtype = np.float)


    # TODO: work for only stochastic policy
    # while True:
    #     delta = 0   # maximum of value function change in each iteration
    #     for state in xrange(env.nS):  # go through all states
    #         new_state_value = 0
    #         for action in xrange(env.nA):   # go through all actions

    #             feedback = env.P[state][action]   # when policy is stochastic, feedback has multiple elements in the list
    #             average_reward = 0    # average reward received by taking a action and blown by the environment
    #             for interaction in xrange(len(feedback)):
    #                 tuple_temp = feedback[interaction]
    #                 # inside average sum in the bellman equation
    #                 # probability * (immediate reward + gamma * state value in the next state)
    #                 average_reward += tuple_temp[0]*(tuple_temp[2] + gamma * value_function[tuple_temp[1]])
                
    #             # outside average sum in the bellman equation
    #             new_state_value += policy[action] * average_reward
    #         delta = max(delta, abs(new_state_value - value_function[state]))
    #         value_function[state] = new_state_value

    #     total_iteration += 1

    #     # stopping criteria
    #     if total_iteration >= max_iterations:
    #         break
    #     if delta <= tol:
    #         break


    # work only for deterministic policy
    while True:
        delta = 0   # maximum of value function change in each iteration
        for state in xrange(env.nS):  # go through all states
            new_state_value = 0
            action = policy[state]    # map state to action based on deterministic policy
            feedback = env.P[state][action]   # when policy is stochastic, feedback has multiple elements in the list
            average_reward = 0    # average reward received by taking a action and blown by the environment
            for interaction in xrange(len(feedback)):
                tuple_temp = feedback[interaction]
                # inside average sum in the bellman equation
                # probability * (immediate reward + gamma * state value in the next state)
                average_reward += tuple_temp[0]*(tuple_temp[2] + gamma * value_function[tuple_temp[1]])
                
            # outside average sum in the bellman equation
            new_state_value = average_reward
            delta = max(delta, abs(new_state_value - value_function[state]))
            value_function[state] = new_state_value

        total_iteration += 1

        # stopping criteria
        if total_iteration >= max_iterations:
            break
        if delta <= tol:
            break

    return value_function, total_iteration


def value_function_to_policy(env, gamma, value_function):
    """Output action numbers for each state in value_function.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute policy for. Must have nS, nA, and P as
      attributes.
    gamma: float
      Discount factor. Number in range [0, 1)
    value_function: np.ndarray
      Value of each state.

    Returns
    -------
    np.ndarray
      An array of integers. Each integer is the optimal action to take
      in that state according to the environment dynamics and the
      given value function.
    """    

    # here, only deterministic policy is considered
    policy = np.zeros(env.nS, dtype = np.int)
    for state in xrange(env.nS):
        optimal_action = 0   
        optimal_reward = -10000    
        for action in xrange(env.nA):
            feedback = env.P[state][action]   # when policy is stochastic, feedback has multiple elements in the list
            average_reward = 0    # average reward received by taking a action and blown by the environment
            for interaction in xrange(len(feedback)):
                tuple_temp = feedback[interaction]
                # inside average sum in the bellman equation
                average_reward += tuple_temp[0] * (tuple_temp[2] + gamma * value_function[tuple_temp[1]])

            if average_reward > optimal_reward:    # argmax
                optimal_reward = average_reward
                optimal_action = action
                # print("optimal action is changed")

        policy[state] = optimal_action  
        # print(action_names[optimal_action])

    # print_policy(policy, action_names)
    return policy


def improve_policy(env, gamma, value_func, policy):
    """Given a policy and value function improve the policy.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

        Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    value_func: np.ndarray
      Value function for the given policy.
    policy: dict or np.array
      The policy to improve. Maps states to actions.

    Returns
    -------
    bool, np.ndarray
      Returns true if policy changed. Also returns the new policy.
    """

    new_policy = value_function_to_policy(env, gamma, value_func)

    if np.array_equal(new_policy, policy):
        return False, new_policy    # policy is unchanged
    else:
        return True, new_policy     # policy is changed


def policy_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    You should use the improve_policy and evaluate_policy methods to
    implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of policy evaluation iterations.
    """


    policy = np.zeros(env.nS, dtype='int')
    iteration_improvement = 0
    iteration_evaluation = 0

    while True:
        value_func, iteration_evaluation_new = evaluate_policy(env, gamma, policy, max_iterations, tol)   # policy evaluation
        improved, policy = improve_policy(env, gamma, value_func, policy)  # policy improvement
        iteration_improvement += 1
        iteration_evaluation += iteration_evaluation_new
        if improved == False:     # policy is stable
            break

    return policy, value_func, iteration_improvement, iteration_evaluation


def value_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.

    See page 90 (pg 108 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """

    total_iteration = 0
    value_function = np.zeros(env.nS, dtype = np.float)

    # value iteration for optimal value function
    while True:
        delta = 0   # maximum of value function change in each iteration
        for state in xrange(env.nS):  # go through all states
            best_average_reward = -1000
            for action in xrange(env.nA):   # go through all actions

                feedback = env.P[state][action]   # when policy is stochastic, feedback has multiple elements in the list
                average_reward = 0    # average reward received by taking a action and blown by the environment
                for interaction in xrange(len(feedback)):
                    tuple_temp = feedback[interaction]
                    # inside average sum in the bellman equation
                    # probability * (immediate reward + gamma * state value in the next state)
                    average_reward += tuple_temp[0]*(tuple_temp[2] + gamma * value_function[tuple_temp[1]])
                
                # find the maximum state-action value 
                if average_reward > best_average_reward:
                    best_average_reward = average_reward

            delta = max(delta, abs(best_average_reward - value_function[state]))
            value_function[state] = best_average_reward     # argmax for optimal value function

        total_iteration += 1
        # stopping criteria
        if total_iteration >= max_iterations:
            break
        if delta <= tol:
            break

    return value_function, total_iteration


def print_policy(policy, action_names, grid_width):
    """Print the policy in human-readable format.

    Parameters
    ----------
    policy: np.ndarray
      Array of state to action number mappings
    action_names: dict
      Mapping of action numbers to characters representing the action.
    """
    str_policy = policy.astype('str')
    for action_num, action_name in action_names.items():  # TODO: mapping check
        np.place(str_policy, policy == action_num, action_name)

    row = []
    delimiter = '\t'
    for state in xrange(len(policy)):
        row.append(str_policy[state])
        if np.mod(state + 1, grid_width) == 0:
            print(delimiter.join(row))
            row = []
            print(" ")


def print_value_function(value_func, grid_width):
    """Print the value function in human-readable format.

    Parameters
    ----------
    value_func: np.ndarray
      Array of state to state value mappings
    grid_width: int
      width of the map: eg, 4 (4x4 environment), 8 (8x8 environment)
    """

    delimiter = '\t'
    row = []
    for state in xrange(len(value_func)):
        row.append("%.5f" % value_func[state])
        if np.mod(state + 1, grid_width) == 0:
            print(delimiter.join(row))
            row = []
            print(" ")
