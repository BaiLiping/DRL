from value_estimator import ValueEstimator
from policy_estimator import PolicyEstimator
from env import PhasedArrayEnv
import numpy as np


def Agent():
    global state
    global td_error
    td_error=0
    global td_target
    td_target=0

    def __init__(self,env,num_episodes,num_steps,discount_rate,learning_rate_1,learning_rate_2,state,td_error):
        self.env=env
        self.num_episodes=num_episodes
        self.num_steps=num_steps
        self.discount_rate=discount_rate
        self.learning_rate_1=learning_rate_1
        self.learning_rate_2=learning_rate_2
        self.state=state
        self.td_error=td_error
        self.td_target=td_target


    def ac_training(self,state):
        record_reward=[]
        record_td_error=[]
        policy_estimator=policy_estimator()
        value_estimator=value_estimator()
        for n in range(self.num_episodes):
            state=env.reset()
            for t in range(self.num_steps):
                prob=policy_estimator.predict(state)
                action=np.random.choice(np.arange(nA),p=prob)
                new_state,reward=env.step(action)
                record_reward.append(reward)
                #td_update after one step
                td_target=reward+discount_rate*(value_estimator.predict(new_state))
                td_error=td_target-value_estimator.predict(state)
                record_td_error.append(td_error)

                policy_estimator.train(state,td_error)
                value_tesimator.train(state,td_target) #probably can feed back error, depending on the implementation
                state=new_state
        
        return record_td_error, policy_estimator, value_estimator




    #def sarsa():
    #def tc():
    #def mc():
    
