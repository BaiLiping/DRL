import gym
import matplotlib
import numpy as np
import sys
import plotting

from collections import defaultdict

from blackjack import BlackjackEnv

matplotlib.style.use('ggplot')

def batch_data(policy, env, num_episodes):
    #generate and memorize a batch of experiences
    action_memory=[]
    state_memory=[]
    reward_memory=[]
    for i in range(num_episodes+1):
        state=env.reset()
        for t in range(100):
            action=policy(state)
            new_state,reward,done,_=env.step(action)
            action_memory.append(action)
            state_memory.append(state)
            reward_memory.append(reward)
            if done:
                break
            state=new_state
    
    print(state_memory,reward_memory)
    return state_memory, reward_memory


    #readback the state memory and update state value to the ultimate score
    #there are few variations, first time you see a state, update it, ignore the other times
    #aother one is everytime you see the state, increamenting it
    #because we are just doing valuation/prediction not control, action is not used
def MC(discount_rate,learning_rate,state_memory, reward_memory):
    V = defaultdict(float)
    for i in range(len(state_memory)):
        if state_memory[i] in V:
            temp=V[state_memory[i]]
            gain=compute_gain_from_t(len(state_memory),reward_memory,i,discount_rate)
            V[state_memory[i]]=temp+learning_rate*(gain-temp)
        else:
            V[state_memory[i]]=compute_gain_from_t(len(state_memory),reward_memory,i,discount_rate)

    return V

def compute_gain_from_t(length,reward_memory,t,discount_rate):
    rate=1
    gain=0
    for i in range(length-t):
        gain+=rate*reward_memory[i+t-1]
        rate*=discount_rate
    return gain

def sample_policy(observation):
    """
    A policy that sticks if the player score is >= 20 and hits otherwise.
    """
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1



if __name__=='__main__':
    env=BlackjackEnv()
    state_memory,reward_memory=batch_data(sample_policy,env,10000)
    V=MC(0.9,0.1,state_memory,reward_memory)
    print(V)
