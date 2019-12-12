import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple

np.set_printoptions(precision=3,suppress=1)
plt.style.use('seaborn-notebook')


class BernoulliBandit(object):
    def __init__(self,success_probabilities, success_reward=1.,fail_reward=0.):
        self._probs=success_probabilities #the leading _ indicate that this is a private variable
        self._number_of_arms=len(self._probs)
        self._s=success_reward
        self._f=fail_reward
        ps=np.array(success_probabilities)
        self._value=ps*success_reward+(1-ps)*fail_reward

    def step(self,action):
        if action<0 or action >=self._number_of_arms:
            raise ValueError('Action {} is out of bounds for a {} armed bandit'.format(action, self._number_of_arms))
        success=bool(np.random.random()<self._probs[action])
        reward=success*self._s+(not success) * self._f
        return reward
    def regret(self,action):
        return self._value.max()-self._value[action]

    def optimal_value(self):
        return self._value.max()

