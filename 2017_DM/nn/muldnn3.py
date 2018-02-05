# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf

# 生成一个先入先出队列和一个QueueRunner
# filenames = ['data/20150401_test.csv']
# filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
# # 定义Reader
# reader = tf.TextLineReader()
# key, value = reader.read(filename_queue)
# # 定义Decoder
# record_defaults = [[1], [1], [1], [1], [1], [1], [1], [1], [1]]
# line, count, isBusytime, isWorkday, isIn, isOK, hour, min, label = tf.decode_csv(value, record_defaults=record_defaults)
# features = tf.stack([line, count, isBusytime, isWorkday, isIn, isOK, hour, min])
# label = label
#
# print features.shape
# print label.shape
#
# # 运行Graph
# with tf.Session() as sess:
#     coord = tf.train.Coordinator()  #创建一个协调器，管理线程
#     threads = tf.train.start_queue_runners(coord=coord)  #启动QueueRunner, 此时文件名队列已经进队。
#     for i in range(10):
#         print features.eval()   #取样本的时候，一个Reader先从文件名队列中取出文件名，读出数据，Decoder解析后进入样本队列。
#         print label.eval()
#     coord.request_stop()
#     coord.join(threads)

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
    file_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs,shuffle=True)
    example, label = read_data(file_queue)

    min_after_dequeue = 10000
    capacity = min_after_dequeue + batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue
    )
    # print example_batch.shape
    return example_batch, label_batch


# x_train_batch, y_train_batch = create_pipeline('data/20150401_2.csv', 1000, num_epochs=1000)
# x_train_batch, y_train_batch = create_pipeline('data/20150401_2.csv', 1000, num_epochs=1000) #514184 line
# x_test, y_test = create_pipeline('data/20150401_1.csv', 200) # 10000行
x_train_batch, y_train_batch = create_pipeline('data/201504_data_train.csv', 1000, num_epochs=1000) #514184 line
x_test, y_test = create_pipeline('data/201504_data_test.csv', 100) # 10000行
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
    print curr_x_test_batch.shape
    while not coord.should_stop():
        if count > 10000:
            break

        # Specify that all features have real-value data
        feature_columns = [tf.contrib.layers.real_valued_column("", dimension=8)]

        # Build 3 layer DNN with 10, 20, 10 units respectively.
        classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
            hidden_units=[10, 20, 10], n_classes=5, model_dir="model/muldnn")


        curr_x_train_batch, curr_y_train_batch = sess.run([x_train_batch, y_train_batch])
        print  curr_x_train_batch.shape

        sess.run(train_step, feed_dict={
            x: curr_x_train_batch,
            y: curr_y_train_batch
        })
        # Define the training inputs
        def get_train_inputs():
            x = tf.constant(curr_x_train_batch)
            y = tf.constant(curr_y_train_batch)
            return x, y

        # Fit model.
        classifier.fit(input_fn=get_train_inputs, steps=2000)

        # Define the test inputs
        def get_test_inputs():
            x = tf.constant(curr_x_test_batch)
            y = tf.constant(curr_y_test_batch)
            return x, y

        # Evaluate accuracy.
        # print(classifier.evaluate(input_fn=get_test_inputs, steps=1))
        accuracy_score = classifier.evaluate(input_fn=get_test_inputs, steps=1)["accuracy"]

        print("nTest Accuracy: {0:f} n".format(accuracy_score))

        # Classify two new flower samples.
        def new_samples():
            return np.array([[1,104,1,1,1,0,8,29], [1,138,1,1,1,1,8,30]], dtype=np.float32)
        # 1, 104, 1, 1, 1, 0, 8, 29, 2
        # 1, 138, 1, 1, 1, 1, 8, 30, 2
        predictions = list(classifier.predict(input_fn=new_samples))

        print("New Samples, Class Predictions:    {} n".format(predictions))

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    # When done, ask the threads to stop.
    coord.request_stop()

# Wait for threads to finish.
coord.join(threads)
sess.close()
