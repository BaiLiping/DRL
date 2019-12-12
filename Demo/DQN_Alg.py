import tensorflow as tf
import numpy as np
import gym

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000
MEMORY_COUNTER = 0          # for store experience
LEARNING_STEP_COUNTER = 0   # for target updating
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
MEMORY = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
class DQN_Alg():
    def define_graph():
        # tf placeholders
        tf_s = tf.placeholder(tf.float32, [None, N_STATES])
        tf_a = tf.placeholder(tf.int32, [None, ])
        tf_r = tf.placeholder(tf.float32, [None, ])
        tf_s_ = tf.placeholder(tf.float32, [None, N_STATES])
        
        with tf.variable_scope('q'):        # evaluation network
            l_eval = tf.layers.dense(tf_s, 10, tf.nn.relu, kernel_initializer=tf.random_normal_initializer(0, 0.1))
            q = tf.layers.dense(l_eval, N_ACTIONS, kernel_initializer=tf.random_normal_initializer(0, 0.1))
        
        with tf.variable_scope('q_next'):   # target network, not to train
            l_target = tf.layers.dense(tf_s_, 10, tf.nn.relu, trainable=False)
            q_next = tf.layers.dense(l_target, N_ACTIONS, trainable=False)
        
        q_target = tf_r + GAMMA * tf.reduce_max(q_next, axis=1)                   # shape=(None, ),
        
        a_indices = tf.stack([tf.range(tf.shape(tf_a)[0], dtype=tf.int32), tf_a], axis=1)
        q_wrt_a = tf.gather_nd(params=q, indices=a_indices)     # shape=(None, ), q for current state
        
        loss = tf.reduce_mean(tf.squared_difference(q_target, q_wrt_a))
        train_op = tf.train.AdamOptimizer(LR).minimize(loss)
    
    def run_graph():
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
    
    
    def choose_action(s):
        s = s[np.newaxis, :]
        if np.random.uniform() < EPSILON:
            # forward feed the observation and get q value for every actions
            actions_value = sess.run(q, feed_dict={tf_s: s})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, N_ACTIONS)
        return action
    
    
    def store_transition(s, a, r, s_):
        global MEMORY_COUNTER
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = MEMORY_COUNTER % MEMORY_CAPACITY
        MEMORY[index, :] = transition
        MEMORY_COUNTER += 1
    
    
    def learn():
        # update target net
        global LEARNING_STEP_COUNTER
        if LEARNING_STEP_COUNTER % TARGET_REPLACE_ITER == 0:
            t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_next')
            e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q')
            sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])
        LEARNING_STEP_COUNTER += 1
    
        # learning
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = MEMORY[sample_index, :]
        b_s = b_memory[:, :N_STATES]
        b_a = b_memory[:, N_STATES].astype(int)
        b_r = b_memory[:, N_STATES+1]
        b_s_ = b_memory[:, -N_STATES:]
        sess.run(train_op, {tf_s: b_s, tf_a: b_a, tf_r: b_r, tf_s_: b_s_})
    
