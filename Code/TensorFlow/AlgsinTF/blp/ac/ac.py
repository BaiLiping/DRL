import gym
import numpy as np
import tensorflow as tf

from cliff_walking import CliffWalkingEnv
from policy_estimator import PolicyEstimator
from value_estimator import ValueEstimator

def actor_critic(env,value_estimator,policy_estimator,num_episode,discount_rate):
    for n in range(num_episode):
        state=env.reset()
        state_memory[]
        action_memory[]
        reward_memory[]

        state_memory.append(state)

        for t in range(1000):
            action_prob=policy_estimator(state)
            action=np.random.choice(np.arange(nA),p=action_prob)
            new_state,reward,done,_=env.step(action)

            state_memory.append(new_state)
            action_memory.append(action)
            reward_memory.append(reward)

            #TD update after every step
            value_prediction=value_estimator.predict(new_state)
            td_target=reward+discount_rate*value_prediction
            td_error=td_target-value_estimator.predict(state)

            value_estimator.update(state,td_target)
            policy_estimator.update(state,td_error,action)

            if done:
                break

            state=new_state
        #record statistics for the plot, after every episode

 

if __name__=="__main__":
    env=CliffWalkingEnv()
    actor_critic(env,.........)
    plot()
