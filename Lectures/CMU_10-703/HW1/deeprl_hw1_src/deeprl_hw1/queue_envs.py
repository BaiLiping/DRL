# coding: utf-8
"""Define the Queue environment from problem 3 here."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from gym import Env, spaces
from gym.envs.registration import register
from gym.utils import seeding
import numpy as np
import copy

# NOTE:
# service happens first before items arrive

# prob_n = [prob_false, prob_true]
def categorical_sample(prob_n):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np.random.rand()).argmax()


class QueueEnv(Env):
    """Implement the Queue environment from problem 3.

    Parameters
    ----------
    p1: float
      Value between [0, 1]. The probability of queue 1 receiving a new item.
    p2: float
      Value between [0, 1]. The probability of queue 2 receiving a new item.
    p3: float
      Value between [0, 1]. The probability of queue 3 receiving a new item.

    Attributes
    ----------
    nS: number of states
    nA: number of actions
    P: environment model
    """
    metadata = {'render.modes': ['human']}

    # action space
    SWITCH_TO_1 = 0
    SWITCH_TO_2 = 1
    SWITCH_TO_3 = 2
    SERVICE_QUEUE = 3

    def __init__(self, p1, p2, p3):     # TODO: check how p1, p2, p3 works
        self.nS = 648
        self.nA = 4
        
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.MultiDiscrete(
            [(1, 3), (0, 5), (0, 5), (0, 5)])   # state space

        # self.P = dict()   #
        self.prob = [p1, p2, p3]
        self.lastaction = None
        self.lastreward = 0
        # self._seed()
        self.s = self._reset() 

    def _reset(self):
        """Reset the environment.

        The server should always start on Queue 1.
        
        Returns
        -------
        (int, int, int, int)
          A tuple representing the current state with meanings
          (current queue, num items in 1, num items in 2, num items in
          3).
        """

        state_tuple = (1, 0, 0, 0)    

        return state_tuple

    def _step(self, action):
        """Execute the specified action.

        Parameters
        ----------
        action: int
          A number in range [0, 3]. Represents the action.

        Returns
        -------
        (state, reward, is_terminal, debug_info)
          State is the tuple in the same format as the reset
          method. Reward is a floating point number. is_terminal is a
          boolean representing if the new state is a terminal
          state. debug_info is a dictionary. You can fill debug_info
          with any additional information you deem useful.
        """

        transitions = self.query_model(self.s, action)
        i = categorical_sample([t[0] for t in transitions])
        prob, nextstate, reward, is_terminal = transitions[i]
        self.s = nextstate
        self.lastaction = action
        self.lastreward = reward
        debug_info = {'prob': prob}
        return (nextstate, reward, is_terminal, debug_info)

    def _render(self, mode='human', close=False):
        print("action is %s" % self.get_action_name(self.lastaction))
        print("reward is %d" % self.lastreward)
        print("current state:")
        print(self.s) 
        print(" ")

    def _seed(self, seed=None):
        """Set the random seed.

        Parameters
        ----------
        seed: int, None
          Random seed used by numpy.random and random.
        """
        
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def query_model(self, state, action):
        """Return the possible transition outcomes for a state-action pair.

        This should be in the same format at the provided environments
        in section 2.

        Parameters
        ----------
        state
          State used in query. Should be in the same format at
          the states returned by reset and step.
        action: int
          The action used in query.
        
        Returns
        -------
        [(prob, nextstate, reward, is_terminal), ...]
          List of possible outcomes
        """
        queue1 = [1 - self.prob[0], self.prob[0]]
        queue2 = [1 - self.prob[1], self.prob[1]]
        queue3 = [1 - self.prob[2], self.prob[2]]
        queue = [queue1, queue2, queue3]
        is_terminal = False
        current_queue = state[0]
        nextstate = copy.copy(list(state))
        reward = 0
        if self.get_action_name(action) == 'SERVICE_QUEUE':
            if state[current_queue] > 0:
                reward = 1
                nextstate[current_queue] -= 1
        elif self.get_action_name(action) == 'SWITCH_TO_1':
            nextstate[0] = 1
        elif self.get_action_name(action) == 'SWITCH_TO_2':
            nextstate[0] = 2
        elif self.get_action_name(action) == 'SWITCH_TO_3':
            nextstate[0] = 3

        # prob [don't add, add]
        prob_list = {}
        for i in xrange(3):
            if nextstate[i+1] == 5:
                prob_list[i] = [1.0, 0]
            else:
                prob_list[i] = queue[i]

        # print('next state is ')
        # print(nextstate)
        
        ret = list()
        for x in xrange(2):
            for y in xrange(2):
                for z in xrange(2):
                    prob_cur = prob_list[0][x] * prob_list[1][y] * prob_list[2][z]
                    if prob_cur == 0:
                        continue
                    else:
                        state_tmp = copy.copy(nextstate)
                        # print(nextstate)
                        state_tmp[1] += x
                        state_tmp[2] += y
                        state_tmp[3] += z
                        # print('x is %d' % x)
                        # print('y is %d' % y)
                        # print('z is %d' % z)
                        tuple_tmp = (prob_cur, tuple(state_tmp), reward, is_terminal)
                        # print(state_tmp)
                        ret.append(tuple_tmp)
        return ret

    def get_action_name(self, action):
        if action == QueueEnv.SERVICE_QUEUE:
            return 'SERVICE_QUEUE'
        elif action == QueueEnv.SWITCH_TO_1:
            return 'SWITCH_TO_1'
        elif action == QueueEnv.SWITCH_TO_2:
            return 'SWITCH_TO_2'
        elif action == QueueEnv.SWITCH_TO_3:
            return 'SWITCH_TO_3'
        return 'UNKNOWN'


register(
    id='Queue-1-v0',
    entry_point='deeprl_hw1.queue_envs:QueueEnv',
    kwargs={'p1': .1,
            'p2': .9,
            'p3': .1})

register(
    id='Queue-2-v0',
    entry_point='deeprl_hw1.queue_envs:QueueEnv',
    kwargs={'p1': .1,
            'p2': .1,
            'p3': .1})
