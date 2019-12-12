import tensorflow as tf
import numpy as np

N,D_in,H,D_out=64,1000,100,10

x=tf.placeholder(tf.float32,shape=(None,D_in))
y=tf.placeholder(tf.float32,shape=(None,D_out))

w1=tf.Variable(tf.random_normal((D_in,H)))
w2=tf.Variable(tf.random_normal((H,D_out)))

h=tf.matmul(x,w1)
h_relu=tf.maximum(h,tf.zeros(1))
y_pred=tf.matmul(h_relu,w2)

loss=tf.reduce_sum((y-y_pred)**2)

grad_w1,grad_w2=tf.gradients(loss,[w1,w2])

learning_rate=1e-6
w1_update=w1.assign(w1-learning_rate*grad_w1)
w2_update=w2.assign(w2-learning_rate*grad_w2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    x_value=np.random.randn(N,D_in)
    y_value=np.random.randn(N,D_out)

    for episode in range(500):
        loss_value,_,_=sess.run([loss,w1_update,w2_update],{x:x_value,y:y_value})
        print(episode,loss_value)
