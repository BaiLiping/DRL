import numpy as np
import MDP
import RL


''' Construct simple MDP as described in Lecture 2a Slides 13-14'''
T = np.array([[[0.5,0.5,0,0],[0,1,0,0],[0.5,0.5,0,0],[0,1,0,0]],[[1,0,0,0],[0.5,0,0,0.5],[0.5,0,0.5,0],[0,0,0.5,0.5]]])
R = np.array([[0,0,10,10],[0,0,10,10]])
discount = 0.9        
mdp = MDP.MDP(T,R,discount)
rlProblem = RL.RL(mdp,np.random.normal)

# Test Q-learning 
Reward_all_trials = np.zeros([100,1000])
for trial in range(0,100):
    [Q,policy,r] = rlProblem.qLearning(s0=0,initialQ=np.zeros([mdp.nActions,mdp.nStates]),nEpisodes=1000,nSteps=100,epsilon=0.3)
    Reward_all_trials[trial,:] = r
    print ("trial", trial)