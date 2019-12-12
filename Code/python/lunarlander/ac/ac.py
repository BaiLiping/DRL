import numpy as np
from keras import backend as k
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam

class Agent(object):
    def __init__(self, alpha,beta,gamma=0.9,n_actions=4,layer1_size=1024,layer2_size=512,input_dims=8):
        self.gamma=gamma
        self.alpha=alpha
        self.beta=beta
        self.input_dims=input_dims
        self.fc1_dims=layer1_size
        self.fc2_dims=layer2_size
        self.n_actions=n_actions
        self.actor,self.critic, self.policy=self.build_actor_critic_network()
        self.action_space=[i for i in range(self.n_actions)]

    
    def build_actor_critic_network(self):
        input=Input(shape=(self.input_dims,))
        delta=Input(shape=[1])
        dense1=Dense(self.fc1_dims,activation='relu')(input)
        dense2=Dense(self.fc2_dims,activation='relu')(dense1)
        probability=Dense(self.n_actions, activation='softmax')(dense2)
        values=Dense(1,activation='linear')(dense2)
         
        def custom_loss(y_true, y_prediction):
            out=k.clip(y_prediction,1e-8,1-1e-8)
            log_likelihood=y_true*k.log(out)
            return k.sum(-log_likelihood*delta)
        actor=Model(input=[input,delta],output=[probability])
        actor.compile(optimizer=Adam(lr=self.alpha),loss=custom_loss)
        critic=Model(input=[input],output=[values])
        critic.compile(optimizer=Adam(lr=self.beta),loss='mean_squared_error')

        policy=Model(input=[input],output=[probability])
        return actor, critic, policy
    



    def choose_action(self, observation):
        state=observation[np.newaxis,:]
        probs=self.policy.predict(state)[0]
        action=np.random.choice(self.action_space, p=probs)

        return action

    def learn(self,state, action, reward, new_state, done):
        state=state[np.newaxis,:]
        new_state=new_state[np.newaxis,:]
    
    
        new_critic_value=self.critic.predict(new_state)
        critic_value=self.critic.predict(state)
        
        target=reward+self.gamma*new_critic_value*(1-int(done))
        delta=target-critic_value
    
        actions=np.zeros([1,self.n_actions])
        actions[np.arange(1),action]=1.0
    
        self.actor.fit([state,delta],actions,verbose=0)
        self.critic.fit(state,target,verbose=0)


   


