# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 15:49:35 2018

@author: Lin Daiwei
"""

import numpy as np
import MDP

class RL2:
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

    def sampleSoftmaxPolicy(self,policyParams,state):
        '''Procedure to sample an action from stochastic policy
        pi(a|s) = exp(policyParams(a,s))/[sum_a' exp(policyParams(a',s))])
        This function should be called by reinforce() to selection actions

        Inputs:
        policyParams -- parameters of a softmax policy (|A|x|S| array)
        state -- current state

        Outputs: 
        action -- sampled action
        '''
        exp_prob = np.exp(policyParams[:,state])
        sum_exp_prob = np.sum(exp_prob)
        exp_prob = exp_prob / sum_exp_prob # normalize exp_prob
        cumProb = np.cumsum(exp_prob)
        action = np.where(cumProb >= np.random.rand(1))[0][0]
        
        return action

    def modelBasedRL(self,s0,defaultT,initialR,nEpisodes,nSteps,epsilon=0):
        '''Model-based Reinforcement Learning with epsilon greedy 
        exploration.  This function should use value iteration,
        policy iteration or modified policy iteration to update the policy at each step

        Inputs:
        s0 -- initial state
        defaultT -- default transition function when a state-action pair has not been vsited
        initialR -- initial estimate of the reward function
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random

        Outputs: 
        V -- final value function
        policy -- final policy
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        V = np.zeros(self.mdp.nStates)
        policy = np.zeros(self.mdp.nStates,int)

        T = defaultT
        R = initialR
        n_s_a = np.zeros((self.mdp.nActions,self.mdp.nStates))
        n_s_a_ns = np.zeros((self.mdp.nActions,self.mdp.nStates,self.mdp.nStates)) 
        
        tolerance = 0.01
#        n_s_a = np.zeros((self.mdp.nActions,self.mdp.nStates))
#        n_s_a_ns = np.zeros((self.mdp.nActions,self.mdp.nStates,self.mdp.nStates)) 
        cum_r_history = np.zeros(nEpisodes)
        for e in range(0,nEpisodes):
            state = s0
            
            for step in range(0,nSteps):        
                action = policy[state]
                if np.random.rand(1) < epsilon:
                    action = np.random.randint(0,self.mdp.nActions)
                
                reward, next_state = self.sampleRewardAndNextState(state, action)
                n_s_a[action, state] = n_s_a[action, state] + 1
                n_s_a_ns[action,state,next_state] = n_s_a_ns[action,state,next_state] + 1
                
                T[action,state,:] = n_s_a_ns[action,state,:]/n_s_a[action,state]
                R[action,state] = (reward + R[action,state]*(n_s_a[action,state]-1))/n_s_a[action,state]
                
                state = next_state
                cum_r_history[e] = cum_r_history[e] + reward*np.power(self.mdp.discount,step)
                
                # value iteration
                epsilon = 0
                V = np.amax(R, axis=0)
                
                epsilon = max(np.absolute(V))
                while epsilon > tolerance:
                    V_all_actions = R + self.mdp.discount*T.dot(V)
                    V_new = np.amax(V_all_actions, axis=0)
                    epsilon = max(np.absolute(V_new - V))
                    V = V_new
                
                # extract policy
                policy = np.argmax(R + self.mdp.discount*T.dot(V), axis=0)
        
#        print("T:\n",T)
#        print("R:\n",R)
#        print("(s,a) \n", n_s_a)
        return V, policy, cum_r_history

    def epsilonGreedyBandit(self,nIterations):
        '''Epsilon greedy algorithm for bandits (assume no discount factor)

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
#        epsilon = 0.3
        state = 0
        n_action = np.zeros(self.mdp.nActions)
        empiricalMeans = np.zeros(self.mdp.nActions)
        r_history = np.zeros(nIterations)
        for i in range(0,nIterations):
            epsilon = 1/(i+1)
            if np.random.rand(1) < epsilon:
                action = np.random.randint(0,self.mdp.nActions)
            else:
                action = np.argmax(empiricalMeans)
            [reward, next_state] = self.sampleRewardAndNextState(state, action)
            r_history[i] = reward
            
            empiricalMeans[action] = ( empiricalMeans[action]*n_action[action] + reward )/(n_action[action] + 1) 
            n_action[action] = n_action[action] + 1
            state = next_state
        return empiricalMeans, r_history

    def thompsonSamplingBandit(self,prior,nIterations,k=1):
        '''Thompson sampling algorithm for Bernoulli bandits (assume no discount factor)

        Inputs:
        prior -- initial beta distribution over the average reward of each arm (|A|x2 matrix such that prior[a,0] is the alpha hyperparameter for arm a and prior[a,1] is the beta hyperparameter for arm a)  
        nIterations -- # of arms that are pulled
        k -- # of sampled average rewards

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nActions)
        state = 0
        r_history = np.zeros(nIterations)
        for i in range(0, nIterations):
            empiricalMeans = np.zeros(self.mdp.nActions)
            for _ in range(0,k):
                empiricalMeans = empiricalMeans + np.random.beta(prior[:,0],prior[:,1])
            empiricalMeans = empiricalMeans/k
            
            action = np.argmax(empiricalMeans)
            reward, next_state = self.sampleRewardAndNextState(state,action)
            r_history[i] = reward
            
            # belief update
            if reward == 1:
                prior[action,0] = prior[action,0] + 1
            else:
                prior[action,1] = prior[action,1] + 1
            
        empiricalMeans = np.zeros(self.mdp.nActions)
        for _ in range(0,10):
            empiricalMeans = empiricalMeans + np.random.beta(prior[:,0],prior[:,1])
        empiricalMeans = empiricalMeans/10
        return empiricalMeans, r_history

    def UCBbandit(self,nIterations):
        '''Upper confidence bound algorithm for bandits (assume no discount factor)

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nActions)
        ucb = np.zeros(self.mdp.nActions)
        n_action = np.ones(self.mdp.nActions)
        state = 0
        r_history = np.zeros(nIterations)
        
        for i in range(0, nIterations):
            ucb = empiricalMeans + np.sqrt( 2*np.log(i)/n_action)
            action = np.argmax(ucb)
            
            [reward, next_state] = self.sampleRewardAndNextState(state,action)
            r_history[i] = reward
            empiricalMeans[action] = ( empiricalMeans[action]*n_action[action] + reward )/(n_action[action] + 1)
            n_action[action] = n_action[action] + 1
            state = next_state

        return empiricalMeans, r_history

    def reinforce(self,s0,initialPolicyParams,nEpisodes,nSteps):
        '''reinforce algorithm.  Learn a stochastic policy of the form
        pi(a|s) = exp(policyParams(a,s))/[sum_a' exp(policyParams(a',s))]).
        This function should call the function sampleSoftmaxPolicy(policyParams,state) to select actions

        Inputs:
        s0 -- initial state
        initialPolicyParams -- parameters of the initial policy (array of |A|x|S| entries)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0)
        nSteps -- # of steps per episode

        Outputs: 
        policyParams -- parameters of the final policy (array of |A|x|S| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        policyParams = initialPolicyParams
        state = s0
        n_s_a = np.zeros((self.mdp.nActions,self.mdp.nStates))
        cum_r_history = np.zeros(nEpisodes)
        for e in range(0,nEpisodes):
            state = s0
#            if e%50 == 0: 
#                print("episode "+ str(e))
            
            # Generate sample path according to policy parameters
            sample_path = np.zeros((nSteps,3))
            
            for step in range(0,nSteps):
                action = self.sampleSoftmaxPolicy(policyParams,state)
                reward, next_state = self.sampleRewardAndNextState(state,action)
                sample_path[step,:] = [state, action, reward]
                n_s_a[action,state] = n_s_a[action,state] + 1
                state = next_state
                
                
            # Update 
            for step in range(0,nSteps):
                G = 0
                for t in range(0,nSteps-step):
                    G = G + np.power(self.mdp.discount, t)*sample_path[step+t][2]
                if step == 0:
                    cum_r_history[e] = G
                [step_state, step_action, step_reward] = sample_path[step,:]
                step_state = int(step_state)
                step_action = int(step_action)
                exp_prob = np.exp(policyParams[:,step_state])
                sum_exp_prob = np.sum(exp_prob)
                exp_prob = exp_prob / sum_exp_prob # normalize exp_prob
                
                d_log_pi = np.zeros(self.mdp.nActions)
                for i in range(0,self.mdp.nActions):
                    if i == step_action:
                        d_log_pi[i] = 1 - exp_prob[i]
                    else:
                        d_log_pi[i] = -exp_prob[i]
                
#                alpha = 1/n_s_a[step_action,step_state]
                alpha = 0.1
                policyParams[:, step_state] = policyParams[:,step_state] + alpha*np.power(self.mdp.discount,step)*G*d_log_pi
        
        # obtain final policy 
        exp_prob = np.exp(policyParams)
        policy = np.argmax(exp_prob, axis=0)
            
        return policy, policyParams, cum_r_history   

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
        cum_r_history = np.zeros(nEpisodes)
        for e in range(0,nEpisodes):
            s = s0
            n = np.zeros([self.mdp.nActions, self.mdp.nStates])
            for step in range(0,nSteps):
                # choose action
                if epsilon > 0 and np.random.rand(1) < epsilon:
                    a = np.random.randint(0,self.mdp.nActions)
                else:
                    a = np.argmax(Q[:,s])
                
                if temperature > 0:
                    # Boltzmann exploration
                    boltzman = np.exp(Q[:,s]/temperature)
                    boltzman = boltzman / np.sum(boltzman) # regularize 
                    cumProb_boltzman = np.cumsum(boltzman)
#                    print(cumProb_boltzman)
                    a = np.where(cumProb_boltzman >= np.random.rand(1))[0][0]
                
                [r, next_state] = self.sampleRewardAndNextState(s,a)
                done = next_state == 16
                cum_r_history[e] += r*np.power(self.mdp.discount,step)
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

        return [Q, policy, cum_r_history]