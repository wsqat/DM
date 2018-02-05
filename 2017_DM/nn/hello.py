# -*- coding: utf-8 -*-
# 简单的神经网络训练mnist
# 每训练100次，测试一次，随着训练次数的增加，测试精度也在增加。训练结束后，1W行数据测试的平均精度为91%左右，不是太高，肯定没有CNN高。
import tensorflow as tf
# import tensorflow.examples.tutorials.mnist.input_data
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# (55000, 784)
# (55000, 10)
# (5000, 784)
# (5000, 10)
# (10000, 784)
# (10000, 10)
# print mnist.train.images.shape
# print mnist.train.labels.shape
# print mnist.validation.images.shape
# print mnist.validation.labels.shape
# print mnist.test.images.shape
# print mnist.test.labels.shape

x = tf.placeholder(tf.float32,[None,784])
y_actual = tf.placeholder(tf.float32,shape=[None,10])

w = tf.Variable(tf.zeros([784,10])) #初始化权值 w
b = tf.Variable(tf.zeros([10])) #初始化偏置值 b

y_predict = tf.nn.softmax(tf.matmul(x,w)+b) #加权变化并进行softmax回归，得到预测概率
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_actual*tf.log(y_predict),reduction_indices=1)) #求交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy) #用梯度下降法使得残差最小
correct_prediction = tf.equal(tf.argmax(y_predict,1),tf.argmax(y_actual,1)) #在测试阶段，测试准确度计算
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float")) #多个批次的准确度均值

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for i in range(10000): #训练阶段，迭代1000次
        batch_xs , batch_ys = mnist.train.next_batch(100) #每批次100行数据
        sess.run(train_step,feed_dict={x:batch_xs, y_actual:batch_ys}) #执行训练
        if (i%100==0): #每训练100次，执行一次
            print "accuracy: ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_actual: mnist.test.labels})
