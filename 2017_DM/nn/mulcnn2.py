# -*- coding: utf-8 -*-
# 卷积神经网络训练mnist
# 训练20000次后，再进行测试，测试精度可以达到99%。

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def read_data(file_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)
    # defaults = [[0], [0.], [0.], [0.], [0.], ['']]
    defaults = [[0], [0], [0], [0], [0], [0], [0], [0], [0]]
    # Id,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species = tf.decode_csv(value, defaults)
    line, count, isBusytime, isWorkday, isIn, isOK, hour, min, label = tf.decode_csv(value, defaults)

    # 因为使用的是鸢尾花数据集，这里需要对y值做转换
    # 因为label 从0开始，这里需要对y值做转换
    label = tf.case({
        tf.equal(label, tf.constant(1)): lambda: tf.constant(0),
        tf.equal(label, tf.constant(2)): lambda: tf.constant(1),
        tf.equal(label, tf.constant(3)): lambda: tf.constant(2),
        tf.equal(label, tf.constant(4)): lambda: tf.constant(3),
        tf.equal(label, tf.constant(5)): lambda: tf.constant(4)
    }, lambda: tf.constant(-1), exclusive=True)

    # return tf.stack([SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]), preprocess_op
    return tf.stack([line, count, isBusytime, isWorkday, isIn, isOK, hour, min]), tf.cast(label,tf.float32)

def create_pipeline(filename, batch_size, num_epochs=None):
    file_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
    example, label = read_data(file_queue)

    min_after_dequeue = 1000
    capacity = min_after_dequeue + batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue
    )
    # print example_batch.shape
    return example_batch, label_batch

# 定义四个函数，分别用于初始化权值W，初始化偏置项b, 构建卷积层和构建池化层。
# 初始化权值W
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

# 初始化偏置项b
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

# 构建卷积层
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')  #strides卷积模板移动的步长，padding 边界处理方式

# 构建池化层
def max_pool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

x_train_batch, y_train_batch = create_pipeline('data/201504_data_train.csv', 1000, num_epochs=1000)
x_test, y_test = create_pipeline('data/201504_data_test.csv',100) # 10000行 200

# x = tf.placeholder(tf.float32,[None,784])
# y_actual = tf.placeholder(tf.float32,shape=[None,10])
x = tf.placeholder(tf.float32,[None,8])
y_actual = tf.placeholder(tf.float32,shape=[None,5])

# 接下来构建网络。整个网络由两个卷积层（包含激活层和池化层），一个全连接层，一个dropout层和一个softmax层组成。
# 构建网络
# x_image = tf.reshape(x,[-1,28,28,1]) #转换输入数据shape,以便于用于网络中
# W_conv1 = weight_variable([5,5,1,32])
# b_conv1 = bias_variable([32])
x_image = tf.reshape(x,[-1,2,4,1]) #转换输入数据shape,以便于用于网络中
W_conv1 = weight_variable([1,1,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)   #第一个卷积层
h_pool1 = max_pool(h_conv1)                             #第一个池化层

# W_conv2 = weight_variable([5,5,32,64])
# W_conv2 = weight_variable([1,1,32,64])
# b_conv2 = bias_variable([64])
# h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)   #第二个卷积层
# h_pool2 = max_pool(h_conv2)                             #第二个池化层

W_fc1 = weight_variable([1*2*32,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool1,[-1,1*2*32])          #reshape成向量
# h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])          #reshape成向量
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1) #第一个全连接层

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)             #dropout层

# W_fc2 = weight_variable([1024,10])
# b_fc2 = bias_variable([10])
W_fc2 = weight_variable([1024,5])
b_fc2 = bias_variable([5])
y_predict = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2) #softmax层

# 训练模型
cross_entropy = -tf.reduce_sum(y_actual * tf.log(y_predict)) #求交叉熵
# train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy) #用梯度下降法使得残差最小
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)  # 梯度下降法
correct_prediction = tf.equal(tf.argmax(y_predict,1),tf.argmax(y_actual,1)) #在测试阶段，测试准确度计算
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float")) #多个批次的准确度均值
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
for i in range(20000): #训练阶段，迭代1000次
    batch = []
    batch[0], batch[1] = sess.run([x_train_batch, y_train_batch])
    # batch = mnist.train.next_batch(50) #每批次50行数据
    if (i%100==0): #每训练100次，执行一次
        # train_acc = accuracy.eval(feed_dict={x:batch[0],y_actual:batch[1],keep_prob:1.0})
        train_acc = accuracy.eval(feed_dict={x: batch[0], y_actual: batch[1], keep_prob: 1.0})
        print 'step %d, training accuracy %g' %(i,train_acc)
        # train_step.run(feed_dict = {x: batch[0], y_actual: batch[1], keep_prob: 0.5})
        train_step.run(feed_dict={x: batch[0], y_actual: batch[1], keep_prob: 0.5})

# test_acc = accuracy.eval(feed_dict={x:mnist.test.images,y_actual:mnist.test.labels,keep_prob:1.0})
x_test, y_test = sess.run([x_test, y_test])
test_acc = accuracy.eval(feed_dict={x:x_test,y_actual:y_test,keep_prob:1.0})
print "test accuracy %g" %test_acc
