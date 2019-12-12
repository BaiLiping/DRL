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
    
    def softmaxPolicyFull(self,policyParams):
        nStates = self.mdp.nStates
        nActions = self.mdp.nActions
        pi = np.zeros((nActions,nStates))
        for state in range(nStates):
            pi[:,state] = self.softmaxPolicy(policyParams,state)
        return pi
    
    def softmaxPolicy(self,policyParams,state):
        x = policyParams[:,state]
        shiftx = x - np.max(x)
        exps = np.exp(shiftx)
        pi = exps / np.sum(exps)
        return pi

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
        
        nActions = self.mdp.nActions
        pi = self.softmaxPolicy(policyParams,state)
        action = np.random.choice(nActions, p=pi)
        return action   

    def epsilonGreedyBandit(self,nIterations):
        '''Epsilon greedy algorithm for bandits (assume no discount factor)

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''
        
        nActions = self.mdp.nActions
        Q = np.zeros(nActions)
        N = np.zeros(nActions)
        state = 0
        rewardsEarned = np.zeros(nIterations)
        for i in range(nIterations):
            epsilon = 1.0 / (i + 1)
            if np.random.rand(1) > epsilon:
                action = np.argmax(Q)
            else:
                action = np.random.randint(nActions)
            #print (action, Q)
            [reward,nextState] = self.sampleRewardAndNextState(state, action)
            rewardsEarned[i] = reward
            state = nextState
            N[action] = N[action] + 1
            Q[action] = Q[action] + 1.0 / N[action] * (reward - Q[action])
            
        empiricalMeans = Q
        return [empiricalMeans, rewardsEarned]

    def thompsonSamplingBandit(self,prior,nIterations,k=1):
        '''Thompson sampling algorithm for Bernoulli bandits (assume no discount factor)

        Inputs:
        prior -- initial beta distribution over the average reward of each arm (|A|x2 matrix such that prior[a,0] is the alpha hyperparameter for arm a and prior[a,1] is the beta hyperparameter for arm a)  
        nIterations -- # of arms that are pulled
        k -- # of sampled average rewards

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''
        
        nActions = self.mdp.nActions
        Q = np.zeros(nActions)
        N = np.zeros(nActions)
        state = 0
        rewardsEarned = np.zeros(nIterations)
        for i in range(nIterations):
            # sample rewards based on prior
            sampleReward = np.zeros(nActions)
            for a in range(nActions):
                [alpha, beta] = prior[a]
                for j in range(k):
                    sampleReward[a] = sampleReward[a] + np.random.beta(alpha,beta)
            sampleReward = sampleReward / k
            # execute action
            action = np.argmax(sampleReward)
            #print (action)
            [reward,nextState] = self.sampleRewardAndNextState(state, action)
            rewardsEarned[i] = reward
            state = nextState
            N[action] = N[action] + 1
            Q[action] = Q[action] + 1.0 / N[action] * (reward - Q[action])
            # update posterior
            if (reward == 1):
                prior[action, 0] = prior[action, 0] + 1
            else:
                prior[action, 1] = prior[action, 1] + 1
        
        empiricalMeans = Q
        return [empiricalMeans, rewardsEarned]

    def UCBbandit(self,nIterations):
        '''Upper confidence bound algorithm for bandits (assume no discount factor)

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        nActions = self.mdp.nActions
        Q = np.zeros(nActions)
        N = np.zeros(nActions)
        state = 0
        rewardsEarned = np.zeros(nIterations)
        for i in range(nIterations):
            UCB = Q + np.sqrt(np.reciprocal(N+1) * 2 * np.log(i+1))
            action = np.argmax(UCB)
            #print (action, Q)
            [reward,nextState] = self.sampleRewardAndNextState(state, action)
            rewardsEarned[i] = reward
            state = nextState
            N[action] = N[action] + 1
            Q[action] = Q[action] + 1.0 / N[action] * (reward - Q[action])
            
        empiricalMeans = Q
        return [empiricalMeans, rewardsEarned]
    
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
        
        nStates = self.mdp.nStates
        nActions = self.mdp.nActions
        V = np.zeros(nStates)
        policy = np.zeros(nStates,int)
        N_sa = np.zeros((nActions,nStates))
        N_sas = np.zeros((nActions,nStates,nStates))
        cumReward = np.zeros(nEpisodes)
        
        T = defaultT
        R = initialR
        discount = self.mdp.discount # assume discount is given
        model = MDP.MDP(T,R,discount)
        
        for episode in range(nEpisodes):
            state = s0
            for step in range(nSteps):
                # epsilon greedy
                if (np.random.rand(1) < epsilon):
                    action = np.random.randint(nActions)
                else:
                    action = policy[state]
                # update T and R
                [reward,nextState] = self.sampleRewardAndNextState(state,action)
                cumReward[episode] = cumReward[episode] + self.mdp.discount**step * reward
                N_sa[action,state] = N_sa[action,state] + 1
                N_sas[action,state,nextState] = N_sas[action,state,nextState] + 1
                model.T[action,state,:] = N_sas[action,state,:].astype(float) / N_sa[action,state]
                model.R[action,state] = model.R[action,state] + 1.0 / N_sa[action,state] * (reward - R[action,state])
                # solve V
                [V,_,_] = model.valueIteration(V)
                policy = model.extractPolicy(V)
                state = nextState
        Q = model.actionValue(V)
        print (Q[:,0:16].reshape((4, 4, 4)))
        return [V,policy,cumReward] 
    
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
        
        nActions = self.mdp.nActions
        nStates = self.mdp.nStates
        discount = self.mdp.discount
        policyParams = initialPolicyParams
        N = np.zeros((nActions,nStates))
        cumReward = np.zeros(nEpisodes)
        
        rewards = np.zeros(nSteps)
        actions = np.zeros(nSteps, dtype=int)
        states = np.zeros(nSteps, dtype=int)
        
        for episode in range(nEpisodes):
            state = s0
            # generate an episode
            for step in range(nSteps):
                action = self.sampleSoftmaxPolicy(policyParams,state)
                [reward,nextState] = self.sampleRewardAndNextState(state,action)
                cumReward[episode] = cumReward[episode] + self.mdp.discount**step * reward
                N[action,state] = N[action,state] + 1
                actions[step] = action
                rewards[step] = reward
                states[step] = state
                state = nextState
            # update policy
            for n in range(nSteps):
                Gn = 0
                for t in range(nSteps - n):
                    Gn = Gn + discount**t * rewards[n+t]
                gradient = -self.softmaxPolicy(policyParams,states[n])
                gradient[actions[n]] = 1 + gradient[actions[n]]
                learningRate = 0.01 # / N[actions[n], states[n]]
                delta = learningRate * discount**n * Gn * gradient
                policyParams[:,states[n]] = policyParams[:,states[n]] + delta
            #print (policyParams)
        return [policyParams, cumReward]
    
    
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
        
        nActions = self.mdp.nActions
        nStates = self.mdp.nStates
        discount = self.mdp.discount
        Q = initialQ
        N = np.zeros((nActions,nStates))
        cumReward = np.zeros(nEpisodes)
        
        for episode in range(nEpisodes):
            state = s0
            for step in range(nSteps):
                # select action
                if (np.random.rand(1) < epsilon):
                    action = np.random.randint(nActions)
                elif (temperature > 0): # TODO not tested!!!
                    p = np.exp(np.divide(Q[:,state], temperature))
                    boltzmann = p / np.sum(p)
                    action = np.random.choice(nActions, p=boltzmann)
                    print ("boltzmann")
                else:
                    action = np.argmax(Q[:,state])
                # go to next state
                [reward,nextState] = self.sampleRewardAndNextState(state,action)
                cumReward[episode] = cumReward[episode] + self.mdp.discount**step * reward
                N[action,state] = N[action,state] + 1
                # update Q
                deltaQ = (reward + discount * Q[:,nextState].max() - Q[action,state])
                learningRate = 1.0 / N[action,state]
                Q[action,state] = Q[action,state] + learningRate * deltaQ
                state = nextState
        #print (N)
        policy = np.argmax(Q, axis=0)
        return [Q,policy, cumReward]
    
    
    def randomWalk(self,s0,nEpisodes,nSteps):
        nActions = self.mdp.nActions
        cumReward = np.zeros(nEpisodes)
        for episode in range(nEpisodes):
            state = s0
            for step in range(nSteps):
                action = np.random.randint(nActions)
                [reward,nextState] = self.sampleRewardAndNextState(state,action)
                cumReward[episode] = cumReward[episode] + self.mdp.discount**step * reward
                state = nextState
        return cumReward
    
    
    
    
    
    
    
    
