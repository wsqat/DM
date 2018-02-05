# -*- coding: utf-8 -*-
#加载包
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# 数据集名称，数据集要放在你的工作目录下
# IRIS_TRAINING = "data/20150401_2.csv"
# IRIS_TEST = "data/20150401_1.csv"

# 数据集读取，训练集和测试集
# training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
#     filename=IRIS_TRAINING,
#     target_dtype=np.int,
#     features_dtype=np.float32)
# test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
#     filename=IRIS_TEST,
#     target_dtype=np.int,
#     features_dtype=np.float32)
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


x_train_batch, y_train_batch = create_pipeline('data/20150401_2.csv', 514184, num_epochs=1000)
# x_train_batch, y_train_batch = create_pipeline('data/20150401_2.csv', 1000, num_epochs=1000) #514184 line
x_test, y_test = create_pipeline('data/20150401_1.csv', 10000) # 10000行

# 特征
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=8)]

# 构建DNN网络，3层，每层分别为10,20,10个节点
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=5,
                                            model_dir="/tmp/iris_model")

# Define the training inputs
def get_train_inputs():
    x = tf.constant(np.array(x_train_batch))
    y = tf.constant(y_train_batch)
    return x, y

# 拟合模型，迭代2000步
classifier.fit(input_fn=get_train_inputs, steps=2000)
# classifier.fit(x=training_set.data,
#                y=training_set.target,
#                steps=2000)

# 计算精度
accuracy_score = classifier.evaluate(x=x_test,y=y_test)["accuracy"]

print('Accuracy: {0:f}'.format(accuracy_score))

# 预测新样本的类别
new_samples = np.array([[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
y = list(classifier.predict(new_samples, as_iterable=True))
print('Predictions: {}'.format(str(y)))