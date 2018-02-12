# coding=utf-8
import tensorflow as tf

print(tf.__version__)

w1 = tf.Variable(tf.random_normal([1, 2], stddev=1, seed=1, mean=10))

# 因为需要重复输入x，而每建一个x就会生成一个结点，计算图的效率会低。所以使用占位符
x = tf.placeholder(tf.float32, shape=(1, 2))
x1 = tf.constant([[0.7, 0.9]])

a = x + w1
b = x1 + w1

sess = tf.Session()
sess.run(tf.global_variables_initializer())


print(sess.run(w1))
# 运行y时将占位符填上，feed_dict为字典，变量名不可变
y_1 = sess.run(a, feed_dict={x: [[0.7, 0.9]]})
y_2 = sess.run(b)
print(y_1)
print(y_2)
sess.close
