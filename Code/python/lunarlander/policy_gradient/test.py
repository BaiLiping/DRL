import gym
import matplotlib.pyplot as plt
import numpy as np
from agent1 import Agent

if __name__=='__main__':
    agent=Agent(ALPHA=0.05,input_dims=8,n_actions=4, layer1_size=64, layer2_size=64)
    env=gym.make('LunarLander-v2')
    score_history=[]
    n_episodes=2000
    for i in range(n_episodes):
        done=False
        score=0
        observation=env.reset()
        while not done:
            aaction=agent.choose_action(observation)
            agent.store_transition(observation,action, reward)
            observation=new_observation
            score+=reward
        score_history.append(score)
        agent.learn()
        print('score{:5f}'.format(score))

