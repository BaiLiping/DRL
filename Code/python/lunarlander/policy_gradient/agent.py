import numpy as np
from keras.layers import Dense, Activation, Input
from keras.models import Model, load_model
from keras.optimizers import Adam
import keras.backend as K

class Agent(object):
    def __init__(self,alpha, gamma=0.99,n_actions=4, layer1_size=16, layer2_size=16, input_dims=128,frame='reinforce.h5'):
        self.alpha=alpha
        self.gamma=gamma
        self.n_actions=n_actions
        self.fc1_dims=layer1_size
        self.fc2_dims=layer2_size
        self.input_dims=input_dims
        self.gain=0
        #memory is needed for monte carlo method TD method doesn't need memory
        self.state_memory=[]
        self.action_memory=[]
        self.reward_memory=[]

        self.policy, self.predict=self.build_policy_network()
        self.action_space=[i for i in range(n_actions)]
        self.model_file=fname

    def build_policy_network(self):
        input=Input(shape=(self.input_dims,))
        advantages=Input(shape=[1])
        dense1=Dense(self.fc1_dims, activation='relu')(input)
        dense2=Dense(self.fc2_dims, activation='relu')(dense1)
        probs=Dense(self.n_actions,activation='softmax')(dense2)

        def custom_loss(y_true,y_predict):
            out=K.clip(y_predict,1e-8,1-1e-8)
            log_lik=y_true*K.log(out)
            return K.sum(-log_lik*advantages)

        policy=Model(input=[input, advantages],output=[probs])
        policy.compile(optimizer=Adam(lr=self.alpha),loss=custom_loss)

        predict=Model(input=[input],output=[probs])
        return policy, predict


    def choose_action(self, observation):
        state=observation[np.newaxis,:]
        probabilities=self.predict.predict(state)[0]
        action=np.random.choice(self.action_space,p=probabilities)
        return action

    def store_transition(self,observation, action, reward):
        self.action_memory.append(action)
        self.state_memory.append(observation)
        self.reward_memory.append(reward)

    def learn(self):
        state_memory=np.array(self.state_memory)
        action_memory=np.array(self.action_memory)
        reward_memory=np.array(self.reward_memory)
        action=np.zeros([len(action_memory),self,n_action])
        action[np.arange(len(action_memory)),action_memory]=1

        gain=np.zeros_like(reward_memory)
        for t in range(len(reward_memory)):
            gain_sum=0
            discount=1
            for k in range(t, len(reward_memory)):
                gain_sum+=rewqard_memory[k]*discount
                dicount*=self.gamma
            gain[t]=gain_sum
        mean=np.mean(gain)
        std=np.std(gain) if np.std(gain)>0 else 1
        self.gain=(gain-mean)/std

        cost=self.policy.train_on_batch([state_memory,self,gain],action)

        self.state_memory=[]
        self.reward_meory=[]
        self.action_memory=[]

    def save_model(self):
        self.policy.save(self.model_file)

    def load_model(self):
        self.policy=load_model(self.model_file)


       




