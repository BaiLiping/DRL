# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 15:17:23 2018

Test convergence of Q-learning for maze problem

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
        print('module folder load successfully')
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

        Q = initialQ
        policy = np.zeros(self.mdp.nStates,int)
        accu_rewards = np.zeros(nEpisodes)
        for ep in range(0,nEpisodes):
            s = s0
            n = np.zeros([self.mdp.nActions, self.mdp.nStates])
            for step in range(0,nSteps):
                # choose action
                if epsilon > 0 and np.random.rand(1) < epsilon:
                    a = np.random.randint(0,self.mdp.nActions)
                else:
                    a = np.argmax(Q[:,s])
                
                [r, next_state] = self.sampleRewardAndNextState(s,a)
                done = next_state == 16
                accu_rewards[ep] += r*np.power(self.mdp.discount,step)
#                if s == 14:
#                    print('r = ', r)
#                    print('next state = ',next_state)
                
                n[a,s] = n[a,s] + 1
                alpha = 1/n[a,s]
                Q[a,s] = Q[a,s] + alpha*(r + self.mdp.discount*np.max(Q[:,next_state]) - Q[a,s]) 
                s = next_state
                
                if done:
                    break
            policy = np.argmax(Q,axis = 0)

        return [Q,policy, accu_rewards]
    