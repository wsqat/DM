# -*- coding:utf-8 -*-
import tensorflow as tf
import sys
from tensorflow.examples.tutorials.mnist import input_data

# this is data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
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

# hyperparameters
# lr = 0.001
lr = 0.1
# training_iters = 100000
training_iters = 10000
batch_size = 128

# x_train_batch, y_train_batch = create_pipeline('data/201504_data_train.csv', batch_size, num_epochs=1000)
x_train_batch, y_train_batch = create_pipeline('data/201504_data_test.csv', batch_size, num_epochs=1000)
# n_inputs = 28  # MNIST data input (img shape: 28*28)
# n_steps = 28  # time steps
# n_hidden_units = 128  # neurons in hidden layer
# n_classes = 10  # MNIST classes (0-9 digits)

n_inputs = 8  # MNIST data input (img shape: 28*28)
n_steps = 1  # time steps
n_hidden_units = 128  # neurons in hidden layer
n_classes = 5  # MNIST classes (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


def RNN(X, weights, biases):
    # hidden layer for input to cell
    ########################################
    # X(128 batch,28 steps,28 inputs)
    # ==>(128*28,28 inputs)
    X = tf.reshape(X, [-1, n_inputs])
    # ==>(128 batch*28 steps,128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # ==>(128 batch,28 steps,128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    # cell
    ##########################################
    # same to define active function
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    # lstm cell is divided into two parts(c_state,m_state)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    # choose rnn how to work,lstm just is one kind of rnn,use lstm_cell for active function,set initial_state
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)

    # hidden layer for output as the final results
    #############################################
    results = tf.matmul(states[1], weights['out']) + biases['out']

    # unpack to list [(batch,outputs)]*steps
    # outputs = tf.unpack(tf.transpose(outputs,[1,0,2])) # state is the last outputs
    # results = tf.matmul(outputs[-1],weights['out']) + biases['out']
    return results


pred = RNN(x, weights, biases)
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    step = 0
    print step * batch_size
    print training_iters
    print "Training "
    while step * batch_size < training_iters:
        print "iterating "
        # batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs, batch_ys = sess.run([x_train_batch, y_train_batch])

        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        # if step % 20 == 0:
        test_acc = sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        print('Step: ', step, 'test accuracy =', test_acc)
        step += 1