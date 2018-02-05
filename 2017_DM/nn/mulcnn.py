# -*- coding:utf-8 -*-
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

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

def read_testset(filename,num_epochs=None):
    file_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
    example, label = read_data(file_queue)

    min_after_dequeue = 1000
    batch_size = 10000
    capacity = min_after_dequeue + batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue
    )
    # print example_batch.shape
    # print example_batch
    return example_batch, label_batch



# shuffle的作用在于指定是否需要随机打乱样本的顺序，一般作用于训练阶段，提高鲁棒性。
# 1、当shuffle = false时，每次dequeue是从队列中按顺序取数据，遵从先入先出的原则
# 2、当shuffle = true时，每次从队列中dequeue取数据时，不再按顺序，而是随机的，所以打乱了样本的原有顺序。
# epochs指的就是训练过程中数据将被“轮”多少次
# 这个参数min_after_dequeue的意思是队列中，做dequeue（取数据）的操作后，queue runner线程要保证队列中至少剩下min_after_dequeue个数据
# 如果min_after_dequeue设置的过少，则即使shuffle为true，也达不到好的混合效果。

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


# x_train_batch, y_train_batch = create_pipeline('Iris-train.csv', 50, num_epochs=1000)
# x_test, y_test = create_pipeline('Iris-20150401_test.csv', 60)
# x_train_batch, y_train_batch = create_pipeline('data/20150401_train.csv', 100, num_epochs=1000)
# x_train_batch, y_train_batch = create_pipeline('data/20150401_train.bak_2.csv', 100, num_epochs=1000)
x_train_batch, y_train_batch = create_pipeline('data/201504_data_train.csv', 1000, num_epochs=1000)
# x_test, y_test = create_pipeline('data/20150401_train.bak_1', 60) # 10000行

# x_test, y_test = create_pipeline('data/20150401_train.bak_1.csv', 1000) # 10000行
# x_test, y_test = read_testset('data/20150401_train.bak_1.csv') # 10000行
x_test, y_test = create_pipeline('data/201504_data_test.csv',100) # 10000行 200
# print x_test.shape
global_step = tf.Variable(0, trainable=False)
learning_rate = 0.1 #tf.train.exponential_decay(0.1, global_step, 100, 0.0)


# Input layer
x = tf.placeholder(tf.float32, [None, 8])
y = tf.placeholder(tf.float32, [None])
# y_actual = tf.placeholder(tf.int32, [None])

# 接下来构建网络。整个网络由两个卷积层（包含激活层和池化层），一个全连接层，一个dropout层和一个softmax层组成。
# 构建网络
x_image = tf.reshape(x,[-1,8,1,1]) #转换输入数据shape,以便于用于网络中
W_conv1 = weight_variable([1,1,1,8])
b_conv1 = bias_variable([8])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)   #第一个卷积层
h_pool1 = max_pool(h_conv1)                             #第一个池化层

# W_conv2 = weight_variable([1,1,8,16])
# b_conv2 = bias_variable([16])
# h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)   #第二个卷积层
# h_pool2 = max_pool(h_conv2)                             #第二个池化层

W_fc1 = weight_variable([1*1*8, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool1,[-1,1*1*8])          #reshape成向量
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1) #第一个全连接层

# h_pool2_flat = tf.reshape(h_pool2,[-1,1*1*16])          #reshape成向量
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1) #第一个全连接层

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)             #dropout层

W_fc2 = weight_variable([1024,5])
b_fc2 = bias_variable([5])
# y_predict = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2) #softmax层
# a = tf.matmul(x, w) + b
a = tf.matmul(h_fc1_drop,W_fc2) + b_fc2
y_predict = tf.nn.softmax(a) #softmax层

# 训练模型
# cross_entropy = -tf.reduce_sum(y_actual * tf.log(y_predict)) #求交叉熵
# cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=a, labels=y_actual))
# # train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy) #用梯度下降法使得残差最小
# train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)  # 梯度下降法
# correct_prediction = tf.equal(tf.argmax(y_predict,1),tf.argmax(y_actual,1)) #在测试阶段，测试准确度计算
# accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float")) #多个批次的准确度均值
# sess = tf.InteractiveSession()
# init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
# sess.run(init)
# for i in range(10000): #训练阶段，迭代1000次
#     # batch = mnist.train.next_batch(50) #每批次50行数据
#     curr_x_train_batch, curr_y_train_batch = sess.run([x_train_batch, y_train_batch])
#     if (i%100==0): #每训练100次，执行一次
#         train_acc = accuracy.eval(feed_dict={x:curr_x_train_batch,y_actual:curr_y_train_batch,keep_prob:1.0})
#         print 'step %d, training accuracy %g' %(i,train_acc)
#         train_step.run(feed_dict = {x: curr_x_train_batch, y_actual: curr_y_train_batch, keep_prob: 0.5})
#
# test_acc = accuracy.eval(feed_dict={x:x_test,y_actual:y_test,keep_prob:1.0})
# print "test accuracy %g" %test_acc


# Training
# cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(a, y))
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=a, labels=tf.cast(y,tf.int64)))
tf.summary.scalar('Cross_Entropy', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.cast(y, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('Accuracy', accuracy)

init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
# merged_summary = tf.summary.merge_all()
merged_summary = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))
sess = tf.Session()
train_writer = tf.summary.FileWriter('logs/train', sess.graph)
test_writer = tf.summary.FileWriter('logs/test', sess.graph)
sess.run(init)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# 保存神经网络模型
saver = tf.train.Saver()

try:
    print("Training: ")
    count = 0
    curr_x_test_batch, curr_y_test_batch = sess.run([x_test, y_test])
    # while not coord.should_stop():
    for i in range(10000): #训练阶段，迭代1000次
        # Run training steps or whatever
        curr_x_train_batch, curr_y_train_batch = sess.run([x_train_batch, y_train_batch])

        sess.run(train_step, feed_dict={
            x: curr_x_train_batch,
            y: curr_y_train_batch
        })

        count += 1
        ce, summary = sess.run([cross_entropy, merged_summary], feed_dict={
            x: curr_x_train_batch,
            y: curr_y_train_batch,
            keep_prob: 1.0
        })

        # train_acc = accuracy.eval(feed_dict={x: curr_x_train_batch, y: curr_y_train_batch, keep_prob: 1.0})

        train_writer.add_summary(summary, count)


        if (i % 100 == 0):  # 每训练100次，执行一次
            ce, test_acc, test_summary = sess.run([cross_entropy, accuracy, merged_summary], feed_dict={
                x: curr_x_test_batch,
                y: curr_y_test_batch,
                keep_prob: 0.5
            })
            test_writer.add_summary(summary, count)
            if test_acc > 0.97:
                saver.save(sess, 'model/mulnn', global_step=i)
            print('Batch', count, 'J  = ', ce, 'test accuracy =', test_acc)
except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    # When done, ask the threads to stop.
    coord.request_stop()

# Wait for threads to finish.
coord.join(threads)
sess.close()

