import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)

x=np.linspace(-1,1,100)[:,np.newaxis]
noise=np.random.normal(0,0.1,size=x.shape)
y=np.power(x,2)+noise

plt.scatter(x,y)
plt.show

tf_x=tf.placeholder(tf.float32,x.shape)
tf_y=tf.placeholder(tf.float32,y.shape)

l1=tf.layers.dense(tf_x,10,tf.nn.relu)
output=tf.layers.dense(l1,1)
#l2=tf.layers.dense(l1,5,tf.nn.sigmoid)
#output=tf.layers.dense(l2,1)

loss=tf.losses.mean_squared_error(tf_y,output)
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.5)
train_op=optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    plt.ion()

    for step in range(100):
        _,l,pred=sess.run([train_op,loss,output],feed_dict={tf_x:x,tf_y:y})
        plt.cla()
        plt.scatter(x,y)
        plt.plot(x,pred,'r')
        plt.text(0.5,0,'Loss={:.4f}'.format(l))
        plt.pause(0.1)

    plt.ioff()
    plt.show()

