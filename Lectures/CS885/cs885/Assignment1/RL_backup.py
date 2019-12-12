# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 22:28:07 2018

@author: Lin Daiwei
"""

import numpy as np
import MDP

class RL:
    def __init__(self,mdp,sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self,state,action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action,state])
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]

    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0):
        '''qLearning algorithm.  Epsilon exploration and Boltzmann exploration
        are combined in one procedure by sampling a random action with 
        probabilty epsilon and performing Boltzmann exploration otherwise.  
        When epsilon and temperature are set to 0, there is no exploration.

        Inputs:
        s0 -- initial state
        initialQ -- initial Q function (|A|x|S| array)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random
        temperature -- parameter that regulates Boltzmann exploration

        Outputs: 
        Q -- final Q function (|A|x|S| array)
        policy -- final policy
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        assert initialQ.shape == (self.mdp.nActions,self.mdp.nStates), "Invalid initial Q shape"
        Q = initialQ
        avg_accu_r = np.zeros(nEpisodes)
        policy = np.zeros(self.mdp.nStates,int)
        n = np.zeros([self.mdp.nActions, self.mdp.nStates])
        for ep in range(0,nEpisodes):
            s = s0
            avg_r = 0
            for step in range(0, nSteps):
                # select an action that has highest Q value
                a = np.argmax(Q[:,s])
                
                if epsilon > 0:
                    # epsilon greedy
                    if np.random.rand(1) < epsilon:
                        a = np.random.randint(0,self.mdp.nActions)
                elif temperature > 0:
                    # Boltzmann exploration
                    boltzman = np.exp(Q[:,s]/temperature)
                    boltzman = boltzman / np.sum(boltzman) # regularize 
                    cumProb_boltzman = np.cumsum(boltzman)
#                    print(cumProb_boltzman)
                    a = np.where(cumProb_boltzman >= np.random.rand(1))[0][0]
                    
                r, next_s = self.sampleRewardAndNextState(s,a)
                done = next_s == 16
                n[a,s] = n[a,s] + 1
                alpha = 1/n[a,s]#learning rate
                Q[a,s] = Q[a,s] + alpha*(r + self.mdp.discount * max(Q[:,next_s])-Q[a,s])
                
                avg_r = (avg_r*step + r)/(step+1)
                s = next_s
                if done: 
                    break
                                
            avg_accu_r[ep] = avg_r
        
        # extract policy
        policy = np.argmax(Q,axis=0)
        return [Q,policy, avg_accu_r]    