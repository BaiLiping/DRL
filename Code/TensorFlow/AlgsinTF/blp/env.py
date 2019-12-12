import gym
import numpy as np
from gym import Env,spaces
from gym.utils import seeding

deck=[1,2,3,4,5,6,7,8,9,10,10,10,10]

def cmp(a,b):
    return int((a>b))-int((a<b))

def draw_card(np_random):
    return np.random.choice(deck)

def draw_hand(np_random):
    return [draw_card(np_random),draw_card(np_random)]

def usable_ace(hand):
    return 1 in hand and sum(hand)<=21

def sum_hand(hand):
    if usable_ace(hand):
        return sum(hand)+10
    return sum(hand)

def is_bust(hand):
    return sum_hand(hand)>21

def score(hand):
    return 0 if is_bust(hand) else sum_hand(hand)

def is_natural(hand):
    return sorted(hand)==[1,10]

class BlackjackEnv(Env):
    def __init__(self, natural=False):
        self.action_space=spaces.Discrete(2)
        self.observation_space=spaces.Tuple((spaces.Discrete(32),spaces.Discrete(11),spaces.Discrete(2)))
        self._seed()

        self.natural=natural
        self._reset()
        self.nA=2


    def reset(self):
        return self._reset()

    def step(self,action):
        return self._step(action)

    def _seed(self,seed=None):
        self.np_random,seed=seeding.np_random(seed)
        return [seed]

    def _step(self,action):
        assert self.action_space.contains(action)
        if action:
            self.player.append(draw_card(self.np_random))
            if is_bust(self.player):
                done=True
                reward=-1
            else:
                done=False
                reward=0
        else:
            done=True
            while sum_hand(self.dealer)<17:
                self.dealer.append(draw_card(self.np_random))
            reward=cmp(socre(self.player),score(self.dealer))
            if self.natural and is_natural(self.player) and reward==1:
                reward=1.5
            
        return self._get_obs(),reward,done,{}#this is a general format for gym environment, new_state, reward, done, additional information

    def _get_obs(self):
        return (sum_hand(self.player),self.dealer[0],usable_ace(self.player))
    #the returned state is sum_hand, dealer's first card, usable_ace

    def _reset(self):
        self.dealer=draw_hand(self.np_random)
        self.player=draw_hand(self.np_random)

        while sum_hand(self.player)<12:
            self.player.append(draw_card(self.np_random))

            return self._get_obs()


if __name__=='__main__':
    env=BlackjackEnv()
    print(env.reset())

