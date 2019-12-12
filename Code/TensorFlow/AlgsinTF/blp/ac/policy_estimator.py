import gym
import numpy as np
import tensorflow as tf

class Policy_Estimator():
    def __init__(self,learning_rate=0.01,scope='policy_estimator'):
        with tf.variable_scope(scope):
            self.state=tf.placeholder(tf.int32,[],'state')
            self.action=tf.placeholder(tf.int32,'action')
            self.target=tf.placeholder(tf.int32,'target')

            state_one_hot=tf.one_hot(self.state,int(env.observation_space.n))
            self.output_layer=tf.contrib.layers.full_connected(inputs=tf.expand_dims(state_one_hot,0),num_output=env.action_space.n,activation_fn=None, weight_initializer=tf.zeros_initializer)

            self.action_prob=tf.squeeze(tf.nn.softmax(self.output_layer))
            self.picked_action_prob=tf.gather(self.action_prob,self.action)

            self.loss=-tf.log(self.picked_action_prob)*self.target

            self.optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op=self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

    def pridict(self.state, sess=None):
        sess=sess or tf.get_default_session()
        return sess.run(self.value_estimate,{self.state:state})


    def update(self,state,target,action, sess=None):
        sess=sess or tf.get_default_session()
        feed_dict={self.state:state,self.target:target,self.action:action}
        _,loss=sess.run([self.train_op,self.loss],feed_dict)
        return loss
