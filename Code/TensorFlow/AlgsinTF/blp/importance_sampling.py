#importance sampling can be thought of as a more sophisticated way of assigning the learning ratio

import gym
import matplotlib
import numpy as np
import sys

import blackjack

def MC_Contro_IS(env, num_episodes,b_policy,discount_rate):
    for i in range(num_episodes+1):
        state=env.reset()
        state_memory[]
        reward_memory[]
        action_memory[]
        state_memory.apend(state)
        for t in range(1000):
            prob=b_policy[state]
            action=np.random.choice(np.arange(nA),p=prob)
            new_state,reward,done,_=env.step(action)
            state_memory.append(new_state)
            reward_memory.append(reward)
            action_memory.append(action)
            if done:
                break
            state=new_state

        one_step_gain=0.0
        W=1.0
        C={}
        Q={}
        for t in range(len(state_memory)-1):
            state=state_memory[t]
            action=action_memory[t]
            reward=reward_memory[t]
            
            W=W*1./target_policy(state)[action]
            greedy_option=argmax(Q[state_memory[t+1]])
            one_step_gain=reward+discount_rate*Q[state_memory[i+1]][greedy_option]
            
            if state in Q:
                if action in Q[state]:
                    C[state][action]+=W
                    temp=Q[state][action]
                    Q[state][action]+=(W/C[state][action])(one_step_gain-temp)
                else:
                    C[state].update({action:W})
                    Q[state].update({action:one_step_gain})
            else:
                C[state]={action:W}
                Q[state]={action:one_step_gain}

            target_policy[state]=argmax(Q[state])

        return Q, target_policy

                    
            


    
