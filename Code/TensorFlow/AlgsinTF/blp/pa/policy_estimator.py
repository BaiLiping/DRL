from keras import backend as K
from keras.layers import Dense, Activation, Input
from keras.model import Model, load_model
from keras.optimizer import Adam
import numpy as np

class PolicyEstimator(object):
    global action
    global state
    global nS
    global nA

    def __init__(self,nS, nA,td_error,state):
        self.nS=nS
        self.layer1_dims=nS
        self.layer2_dims=nS
        self.nA=nA
        self.state=state
        self.action=action
    
        input_layer=Input((nS,))
        hidden_layer_1=Dense(layer1_dims,activation='relu')(input_layer)
        hidden_layer_2=Dense(layer2_dims,activation='relu')(hidden_layer_1)
        probability_distribution=Dense(nA,activation='softmax')(hidden_layer_2)
        policy_estimator=Model([input_layer,delta],[probability_distribution])
    
    def get_probability(self):

        def custom_loss(y_true, y_predict):
            out  = K.clip(y_predict, 1e-8,1-1e-8)
            log_likelihood=y_true*K.log(out)
            return k.sum(-log_likelihood*delta)
        
        policy_estimator.compile(optimizer=Adam(self.Alpha),loss=custom_loss)
        probability=policy_estimator.predict(self.state)

        return probability_distribution
    
    def train(self,td_error):
        self.policy_estimator.fit([state,delta],actions)
