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


x_train_batch, y_train_batch = create_pipeline('data/20150401_2.csv', 1000, num_epochs=1000)
# x_train_batch, y_train_batch = create_pipeline('data/20150401_2.csv', 1000, num_epochs=1000) #514184 line
x_test, y_test = create_pipeline('data/20150401_1.csv', 100) # 10000行
# x_train_batch, y_train_batch = create_pipeline('data/201504_data_train.csv', 1000, num_epochs=1000) #514184 line
# x_test, y_test = create_pipeline('data/201504_data_test.csv', 500) # 10000行

init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
merged_summary = tf.summary.merge_all()

sess = tf.Session()
sess.run(init)


print("Training: ")
count = 0
curr_x_test_batch, curr_y_test_batch = sess.run([x_test, y_test])

# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=8)]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
    hidden_units=[10, 20, 10], n_classes=5, model_dir="model/muldnn")


curr_x_train_batch, curr_y_train_batch = sess.run([x_train_batch, y_train_batch])
print  curr_x_train_batch.shape


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

sess.close()
