import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)

x=np.linspace(-1,1,100)[:,np.newaxis]
noise=np.random.normal(0,0.1,size=x.shape)
y=np.power(x,2)+noise

def save():
    print('this is save')
    tf_x=tf.placeholder(tf.float32,x.shape)
    tf_y=tf.placeholder(tf.float32,y.shape)
    layer1=tf.layers.dense(tf_x,10,tf.nn.relu)
    output=tf.layers.dense(layer1,1)
    loss=tf.losses.mean_squared_error(tf_y,output)
    train_operation=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver=tf.train.Saver()
        for i in range(100):
            output,loss,sess.run([output,loss,train_operation],feed_dict={tf_x:x,tf_y:y})
        saver.save(sess,'./blpparams',write_meta_graph=False)

    plt.figure(1, figsize=(10, 5))
    plt.scatter(x, y)
    plt.plot(x, output, 'r-', lw=5)
    plt.text(-1, 1.2, 'Save Loss=%.4f' % l, fontdict={'size': 15, 'color': 'red'})

def load():
    print('this is load')
    tf_x=tf.placeholder(tf.float32,x.shape)
    tf_y=tf.placeholder(tf.float32,y.shape)

    layer1=tf.layers.dense(tf_x,10,tf.nn.relu)
    ouput=tf.layers.dense(layer1,1)
    loss=tf.losses.mean_squared_error(tf_y,output)

    with tf.Session() as sess:
        saver=tf.train.Saver()
        saver.restore(sess,'./blpparams')

        output, l = sess.run([output, loss], {tf_x: x, tf_y: y})
    plt.subplot(122)
    plt.scatter(x, y)
    plt.plot(x, pred, 'r-', lw=5)
    plt.text(-1, 1.2, 'Reload Loss=%.4f' % l, fontdict={'size': 15, 'color': 'red'})
    plt.show()

save()
tf.reset_defaut_graph()
load()



