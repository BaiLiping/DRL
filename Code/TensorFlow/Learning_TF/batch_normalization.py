import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

n_samples=2000
batch_size=64
epoch=12
learning_rate=0.01
hidden_units=8
activation=tf.nn.relu

x=np.linspace(-7,10,n_samples)[:,np.newaxis]
np.random.shuffle(x)
noise=np.random.normal(0,2,x.shape)
y=np.square(x)-5+noise
train_data=np.hstack((x,y))

test_x=np.linspace(-7,10,200)
noise=np.random.normal(0,2,test_x.shape)
test_y=np.square(test_x)-5+noise


tf_x=tf.placeholder(tf.float32,[None,1])
tf_y=tf.placeholder(tf.float32,[None,1])
tf_is_train=tf.placeholder(tf.bool,None)

class NN(object):
    def __init__(self,batch_normalization=False):
        self.is_bn=batch_normalization
        self.w_init=tf.random_normal_initializer(0.,.1)
        self.data=[tf_x]
        if self.is_bn:
            self.input_layer=[tf.layers.batch_normalization(tf_x,training=tf_is_train)]
        else:
            self.input_layer=self.data
        for i in range(hidden_units):
            self.input_layer.append(self.add_layer())
