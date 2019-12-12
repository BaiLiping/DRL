import tensorflow as tf
x1=tf.placeholder(dtype=tf.float32,shape=None)
y1=tf.placeholder(dtype=tf.float32,shape=None)

z1=x1+y1

x2=tf.placeholder(dtype=tf.float32,shape=[2,1])
y2=tf.placeholder(dtype=tf.float32,shape=[1,2])
z2=tf.matmul(x2,y2)

with tf.Session() as sess:
    z1_value=sess.run(z1,feed_dict={x1:1,y1:4})
    z2_value=sess.run(z2,feed_dict={x2:[[2],[5]],y2:[[4,6]]})
    print(z1_value)
    print(z2_value)
