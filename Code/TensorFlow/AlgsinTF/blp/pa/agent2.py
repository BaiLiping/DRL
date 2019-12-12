from keras import backend as K
from keras.layers import Dense, Activation, Input
from keras.models import Model, load_model
from keras.optimizers import Adam
import numpy as np
from env import PhasedArrayEnv

class Agent(object):
    def __init__(self, alpha, beta, gamma, nA,nS):
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.input_dims = nS
        self.fc1_dims = 1024
        self.fc2_dims = 1024
        self.nA = nA

        self.actor, self.policy = self.policy_estimator()
        self.critic=self.value_estimator()
        self.action_space = [i for i in range(nA)]

    def policy_estimator(self):
        data = Input(shape=(self.input_dims,))
        td_error = Input(shape=[1])
        dense1 = Dense(self.fc1_dims, activation='relu')(data)
        dense2 = Dense(self.fc2_dims, activation='relu')(dense1)
        probs = Dense(self.nA, activation='softmax')(dense2)

        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            log_lik = y_true*K.log(out)
            return K.sum(-log_lik*td_error)

        actor = Model(input=[data, td_error], output=[probs])
        actor.compile(optimizer=Adam(lr=self.alpha), loss=custom_loss)
        distribution = Model(input=[data], output=[probs])
        return actor,distribution
    
    def value_estimator(self):
        data=Input(shape=(self.input_dims,))
        dense1=Dense(self.fc1_dims,activation='relu')(data)
        dense2=Dense(self.fc2_dims,activation='relu')(dense1)
        output=Dense(1, activation='linear')(dense2)
        critic=Model(input=[data], output=[output])
        critic.compile(optimizer=Adam(lr=self.beta), loss='mean_squared_error')
        return critic
    
    def choose_action(self, state):
        probabilities = self.policy.predict(state)
        action = np.random.choice(range(nA), p=probabilities)
        return action

    def learn(self, state, action, reward, new_state):
        new_critic_value = self.critic.predict(new_state)
        critic_value = self.critic.predict(state)
        
        td_target = reward + self.gamma*new_critic_value
        td_error =  td_target - critic_value

        self.actor.fit([state, td_error], actions)
        self.critic.fit(state, td_target)


def main():
    num_episodes=10000
    num_steps=9

    gamma=0.9
    alpha=0.01
    beta=0.01

    env=PhasedArrayEnv()
    nA,nS=env.get_nAnS()

    agent=Agent(alpha,beta,gamma,nA,nS)

    score_history=[]
    for n in range(num_episodes):
        state=env.reset()
        for t in range(num_steps):
            action=agent.choose_action(state)
            new_state,reward=env.step(action)
            agent.learn(state,action,reward,new_state)
            state=new_state
            score+=reward
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        print('episode: ', i,'score: %.2f' % score,
                'avg score %.2f' % avg_score)

if __name__=='__main__':
    main()
