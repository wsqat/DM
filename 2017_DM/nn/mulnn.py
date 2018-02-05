# -*- coding:utf-8 -*-
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
    return tf.stack([line, count, isBusytime, isWorkday, isIn, isOK, hour, min]), label

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

# shuffle的作用在于指定是否需要随机打乱样本的顺序，一般作用于训练阶段，提高鲁棒性。
# 1、当shuffle = false时，每次dequeue是从队列中按顺序取数据，遵从先入先出的原则
# 2、当shuffle = true时，每次从队列中dequeue取数据时，不再按顺序，而是随机的，所以打乱了样本的原有顺序。
# epochs指的就是训练过程中数据将被“轮”多少次
# 这个参数min_after_dequeue的意思是队列中，做dequeue（取数据）的操作后，queue runner线程要保证队列中至少剩下min_after_dequeue个数据
# 如果min_after_dequeue设置的过少，则即使shuffle为true，也达不到好的混合效果。

# x_train_batch, y_train_batch = create_pipeline('Iris-train.csv', 50, num_epochs=1000)
# x_test, y_test = create_pipeline('Iris-20150401_test.csv', 60)
# x_train_batch, y_train_batch = create_pipeline('data/20150401_train.csv', 100, num_epochs=1000)
x_train_batch, y_train_batch = create_pipeline('data/20150401_2.csv', 1000)
# x_train_batch, y_train_batch = create_pipeline('data/201504_data_train.csv', 1000, num_epochs=1000)
# x_test, y_test = create_pipeline('data/20150401_train.bak_1', 60) # 10000行
x_test, y_test = create_pipeline('data/20150401_1.csv', 100) # 10000行
# x_test, y_test = create_pipeline('data/201504_data_test.csv',500) # 10000行 200
# print x_test.shape
global_step = tf.Variable(0, trainable=False)
learning_rate = 0.1#tf.train.exponential_decay(0.1, global_step, 100, 0.0)


# Input layer
x = tf.placeholder(tf.float32, [None, 8])
y = tf.placeholder(tf.int32, [None])

# Output layer
w = tf.Variable(tf.random_normal([8, 5]))
b = tf.Variable(tf.random_normal([5]))
a = tf.matmul(x, w) + b
prediction = tf.nn.softmax(a)

# Training
# cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(a, y))
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=a, labels=y))
tf.summary.scalar('Cross_Entropy', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.cast(y, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('Accuracy', accuracy)

init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
merged_summary = tf.summary.merge_all()

sess = tf.Session()
train_writer = tf.summary.FileWriter('logs/train', sess.graph)
test_writer = tf.summary.FileWriter('logs/test', sess.graph)
sess.run(init)

coord = tf.train.Coordinator()  # 创建一个协调器,管理线程
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# 保存神经网络模型
saver = tf.train.Saver()

try:
    print("Training: ")
    count = 0
    curr_x_test_batch, curr_y_test_batch = sess.run([x_test, y_test])
    # print curr_x_test_batch.shape
    while not coord.should_stop():
        if count > 10000:
            break
    # for i in range(10000): #训练阶段，迭代1000次
        # Run training steps or whatever
        curr_x_train_batch, curr_y_train_batch = sess.run([x_train_batch, y_train_batch])
        # print curr_x_train_batch.shape

        sess.run(train_step, feed_dict={
            x: curr_x_train_batch,
            y: curr_y_train_batch
        })

        count += 1
        ce, summary = sess.run([cross_entropy, merged_summary], feed_dict={
            x: curr_x_train_batch,
            y: curr_y_train_batch
        })

        train_writer.add_summary(summary, count)


        if (count % 100 == 0):  # 每训练100次，执行一次
            ce, test_acc, test_summary = sess.run([cross_entropy, accuracy, merged_summary], feed_dict={
                x: curr_x_test_batch,
                y: curr_y_test_batch
            })
            test_writer.add_summary(summary, count)
            if test_acc > 0.9:
                print "save model: "+str(count)
                saver.save(sess, 'model/mulnn', global_step=count)
            print('Batch', count, 'J = ', ce, 'test accuracy =', test_acc)
except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    # When done, ask the threads to stop.
    coord.request_stop()

# Wait for threads to finish.
coord.join(threads)
sess.close()

