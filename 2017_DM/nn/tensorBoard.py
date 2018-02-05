# -*- coding: utf-8 -*-
import tensorflow as tf

# 定义一个简单的计算图，实现向量加法的操作
input1 = tf.constant([1.0,2.0,3.0],name="input1")
input2 = tf.Variable(tf.random_uniform([3]),name="input2")
output = tf.add_n([input1,input2],name="add")

# writer = tf.train.SummaryWriter("/path/to/log",tf.get_default_graph())
writer = tf.summary.FileWriter("/temp/to/log",tf.get_default_graph())
writer.close()
