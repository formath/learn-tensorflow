#!/usr/bin/env python

import sys
import os
import tensorflow as tf
import numpy as np
from dataset import data_set

class BinaryLogisticRegression(object):
    def __init__(self, feature_num):
        self.feature_num = feature_num
        self.x = tf.placeholder("float", [None, self.feature_num])
        self.w = tf.Variable(tf.random_normal([self.feature_num, 1], stddev=0.1))
        self.y = tf.placeholder("float", [None, 1])

    def forward(self):
        if type(self.x) == np:
            return tf.matmul(self.x, self.w)
        else:
            return tf.nn.embedding_lookup_sparse(self.w, self.x[0], self.x[1], combiner="sum")


# train process
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate')
flags.DEFINE_integer('max_iter', 20, 'Max iteration for training')
flags.DEFINE_integer('batch_size', 500, 'Batch size')
flags.DEFINE_string('train_file', '../data/train.data', 'training data')
flags.DEFINE_string('test_file', '../data/test.data', 'test data')

train_file = FLAGS.train_file
test_file = FLAGS.test_file
learning_rate = FLAGS.learning_rate
max_iter = FLAGS.max_iter
batch_size = FLAGS.batch_size

train_set = data_set.DataSet()
train_set.load(train_file, 100, dense=True)
test_set = data_set.DataSet()
test_set.load(test_file, 100, dense=True)

model = BinaryLogisticRegression(100)
y = model.forward()
loss = tf.reduce_sum(tf.nn.signoid_cross_entropy_with_logits(y, model.y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
probability_output = tf.nn.sigmoid(y)

session = tf.Session()
init_variable = tf.initialize_all_variables()
session.run(init_variable)

for i in range(max_iter):
    while not train_set.one_pass:
        mb_x, mb_y = train_set.mini_batch(batch_size)
        session.run([optimizer, loss], feed_dict={model.x: mb_x, model.y: mb_y})
        print("loss: ", loss)