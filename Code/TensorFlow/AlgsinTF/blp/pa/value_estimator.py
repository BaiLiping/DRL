from keras.layers import Dense, Activation, Input
from keras.model import Model, load_model
from keras.optimizer import Adam
import numpy as np

class ValueEstimator(object):
    def __init__(self,nS,nA):
        self.nS=nS
        self.layer1_dims=nS
        self.layer2_dims=nS
        self.nA=nA
        self.state=state
        self.td_target=td_target

        input_layer=Input((nS,))
        hidden_layer1=Dense(layer1_dims,activation='relu')(input_layer)
        hidden_layer2=Dense(layer2_dims,activation='relu')(hidden_layer1)
        value=0
        value_estimator=Model([input_layer],value,activation='linear')(hidden_layer2)
        
    def get_value(self):
        self.value_estimator.compile(optimizer=Adam(self.beta),loss='mean_squared_loss')
        value=value_estimator.predict(self.state)
        return value

    def train(self):
        self.value_estimator.fit([self.state],td_target)

