import gym
import numpy as np
import sys
from windy_gridworld import WindyGridworldEnv

UP=0
RIGHT=1
DOWN=2
LEFT=3

def greedy_policy(nA,state,Q,epsilon):
    greedy_policy=np.ones(nA,dtype=float)*epsilon/nA
    greedy_action=np.argmax(Q[state])
    greedy_policy[greedy_action]+=(1-epsilon)
    return greedy_policy


def Sarsa(env, num_episodes, discount_rate, learning_rate,epsilon,decay):
    #on-policy TD control
    for n in range(num_episodes):
        state=env.reset()
        for t in range(100):
            prob=greedy_policy[state]
            action=np.random.choice(np.arange(nA),p=prob)
            new_state,reward, done, _ =env.step(action)
            new_prob=greedy_policy[new_state]
            new_action=np.random.choice(np.arange(nA),p=new_prob)

            #after taking one step, update Q
            td_target=reward+discount_rate*Q[new_state][new_action]
            if state in Q:
                if action in Q[state]:
                    temp=Q[state][action]
                    Q[state][action]+=learning_rate*(td_target-temp)
                else:
                    Q[state].update({action:td_target})
            else:
                Q[state]={action:td_target}

            if done:
                break

            state=new_state
            action=new_action

    return Q

if __name__=='__main__':
    env=WindyGridworldEnv()
    

