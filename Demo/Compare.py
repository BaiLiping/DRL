from Algs import algs
import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt
tf.set_random_seed(1)
np.random.seed(1)

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


algs=algs()
with tf.name_scope('a'):
    algs.define_graph()
    #algs.run_graph()
    reward_a_with_w=algs.acceleration_with_weights()
with tf.name_scope('b'):
    algs.define_graph()
    #algs.run_graph()
    reward_a=algs.with_acceleration()
with tf.name_scope('c'):
    algs.define_graph()
    #algs.run_graph()   
    reward_v=algs.vanilla()
x=np.arange(400)
plt.scatter(x,reward_a_with_w,label='acceleration with weights')
plt.scatter(x,reward_a,label='with acceleration')
plt.scatter(x,reward_v,lavel='vanilla')
plt.legend()
plt.show()

