import gym
import matplotlib
import numpy as np
import sys
import plotting

from collections import defaultdict

from blackjack import BlackjackEnv

matplotlib.style.use('ggplot')

def MC_Control(env,nA,num_episodes,epsilon,decay,learning_rate,discount_rate):
    #generate and memorize a batch of experiences
    Q ={}
    policy={}

    for i in range(num_episodes+1):
        epsilon*=decay
        action_memory=[]
        state_memory=[]
        reward_memory=[]
        state=env.reset()
        for t in range(100):
            def epsilon_greedy(state,Q):
                prob=np.ones(nA,dtype=float)*(epsilon/nA)
                q_score=np.zeros(nA)
                if state in Q:
                    for action in range(nA):
                        if action in Q[state]:
                            q_score[action]=Q[state][action]
                        else:
                            q_score[action]=0
                else:
                    for action in range(nA):
                        q_score[action]=0
                greedy_option=np.argmax(q_score)
                prob[greedy_option]+=(1-epsilon)
                return prob

            prob=epsilon_greedy(state,Q)
            policy[state]=prob
            action=np.random.choice(np.arange(len(prob)),p=prob)
            new_state,reward,done,_=env.step(action)
            action_memory.append(action)
            state_memory.append(state)
            reward_memory.append(reward)
            if done:
                break
            state=new_state
        for i in range(len(state_memory)):
            gain=compute_gain_from_t(len(state_memory),reward_memory,i,discount_rate)
            if state_memory[i] in Q:
                if action_memory[i] in Q[state_memory[i]]:
                    temp=Q[state_memory[i]][action_memory[i]]
                    Q[state_memory[i]][action_memory[i]]=temp+learning_rate*(gain-temp)
                else:
                    Q[state_memory[i]].update({action_memory[i]:gain})
            else:
                Q[state_memory[i]]={action_memory[i]:gain}

    return Q,policy

def compute_gain_from_t(length,reward_memory,t,discount_rate):
    rate=1
    gain=0
    for i in range(length-t):
        gain+=rate*reward_memory[i+t-1]
        rate*=discount_rate
    return gain



if __name__=='__main__':
    env=BlackjackEnv()
    nA=env.action_space.n
    Q=MC_Control(env,nA,1000,0.9,0.99,0.1,0.9)
    print(Q)
