import tensorflow as tf

m1=tf.constant([[2,2]])
m2=tf.constant([[3],[3]])

dot_operation=tf.matmul(m1,m2)

sess=tf.Session()
result=sess.run(dot_operation)
print(result)
sess.close()

with tf.Session() as sess:
    result_=sess.run(dot_operation)
    print(result_)

