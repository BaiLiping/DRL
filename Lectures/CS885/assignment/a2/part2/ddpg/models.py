import tensorflow as tf
import tensorflow.contrib as tc


class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        collection = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        shared = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="shared")
        return (collection+shared)

    @property
    def trainable_vars(self):
        collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        shared = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="shared")
        return (collection+shared)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


class Actor(Model):
    def __init__(self, nb_actions, name='actor', layer_norm=True):
        super(Actor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm

    def __call__(self, obs, reuse=False):
        
        x = obs
        x = tf.layers.dense(x, 64, name="shared", reuse=tf.AUTO_REUSE)
        if self.layer_norm:
            x = tc.layers.layer_norm(x, center=True, scale=True, scope="shared", reuse=tf.AUTO_REUSE)
        x = tf.nn.relu(x)
        
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            
            
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            
            x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)
        return x


class Critic(Model):
    def __init__(self, name='critic', layer_norm=True):
        super(Critic, self).__init__(name=name)
        self.layer_norm = layer_norm

    def __call__(self, obs, action, reuse=False):
    
        x = obs
        x = tf.layers.dense(x, 64, name="shared", reuse=True)
        if self.layer_norm:
            x = tc.layers.layer_norm(x, center=True, scale=True, scope="shared", reuse=True)
        x = tf.nn.relu(x)
            
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            
            x = tf.concat([x, action], axis=-1)
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
