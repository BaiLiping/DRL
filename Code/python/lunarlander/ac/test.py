import gym
from ac import Agent
#from utils import plotLearning
import numpy as np


def main():
    agent=Agent(alpha=0.00001,beta=0.00005)
    env=gym.make('LunarLander-v2')
    score_history=[]
    num_episodes=2000

    for i in range(num_episodes):
        done=False
        score=0
        observation=env.reset()
        while not done:
            action=agent.choose_action(observation)
            new_observation,reward,done,info=env.step(action)
            agent.learn(observation, action,reward, new_observation,done)
            observation=new_observation
            score+=score
        score_history.append(score)
        avg_score=np.mean(score_history[-100:])
        print('episode{}: score {:.10f} average score {:.10f}'.format(i,score,avg_score))

        
 #   filename='lunar.png'
 #   plotlearning(score_history,filename=filename,window=100)


if __name__== "__main__":
    main()
