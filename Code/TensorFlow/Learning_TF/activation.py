import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(-5,5,200)
#np.linspace(start, stop, number) evenly generate number of data points evenly distributed between start and end

y_relu=tf.nn.relu(x)
y_sigmoid=tf.nn.sigmoid(x)
y_tanh=tf.nn.tanh(x)
y_softplus=tf.nn.softplus(x)
y_softmax=tf.nn.softmax(x)

with tf.Session() as sess:
    y_relu,y_sigmoid,y_tanh,y_softplus,y_softmax=sess.run([y_relu,y_sigmoid,y_tanh,y_softplus,y_softmax])

plt.figure(1,figsize=(8,6))
plt.subplot(121)
plt.plot(x,y_sigmoid,label='sigmoid')
plt.plot(x,y_relu,label='relu')
plt.plot(x,y_tanh,label='tanh')
plt.plot(x,y_softplus,label='softplus')
plt.plot(x,y_softmax,label='softmax')
plt.legend()
plt.show()

