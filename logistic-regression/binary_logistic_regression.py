#!/usr/bin/env python

import sys
import os
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import numpy as np
from dataset import data_set

class BinaryLogisticRegression(object):
    def __init__(self, feature_num, dense = True):
        self.feature_num = feature_num
        self.dense = dense
        if dense:
            self.x = tf.placeholder("float", [None, self.feature_num])
            self.w = tf.Variable(tf.random_normal([self.feature_num, 1], stddev=0.1))
            self.y = tf.placeholder("float", [None, 1])
        else:
            self.sparse_index = tf.placeholder(tf.int64)
            self.sparse_ids = tf.placeholder(tf.int64)
            self.sparse_values = tf.placeholder(tf.float32)
            self.sparse_shape = tf.placeholder(tf.int64)
            self.w = tf.Variable(tf.random_normal([self.feature_num, 1], stddev=0.1))
            self.y = tf.placeholder("float", [None, 1])

    def forward(self):
        if self.dense:
            return tf.matmul(self.x, self.w)
        else:
            return tf.nn.embedding_lookup_sparse(self.w,
                                                 tf.SparseTensor(self.sparse_index, self.sparse_ids, self.sparse_shape),
                                                 tf.SparseTensor(self.sparse_index, self.sparse_values, self.sparse_shape),
                                                 combiner="sum")


# train process
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.001, 'initial learning rate')
flags.DEFINE_integer('max_iter', 10, 'max iteration for training')
flags.DEFINE_integer('batch_size', 100, 'batch size')
flags.DEFINE_string('train_file', '../data/train.data', 'training data')
flags.DEFINE_string('test_file', '../data/test.data', 'test data')
flags.DEFINE_integer('feature_num', 30, 'feature num')
flags.DEFINE_boolean('dense', True, 'dense feature')

train_file = FLAGS.train_file
test_file = FLAGS.test_file
learning_rate = FLAGS.learning_rate
max_iter = FLAGS.max_iter
batch_size = FLAGS.batch_size
feature_num = FLAGS.feature_num
dense = FLAGS.dense

train_set = data_set.DataSet()
train_set.load(train_file, feature_num, dense)
test_set = data_set.DataSet()
test_set.load(test_file, feature_num, dense)

model = BinaryLogisticRegression(feature_num)
y = model.forward()
loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(y, model.y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
probability_output = tf.nn.sigmoid(y)
#auc = tf.contrib.metrics.streaming_auc(probability_output, model.y, num_thresholds=200)

session = tf.Session()
init_all_variable = tf.initialize_all_variables()
init_local_variable = tf.initialize_local_variables()
session.run([init_all_variable, init_local_variable])

while train_set.epoch_pass < max_iter:
    if model.dense:
        mb_x, mb_y = train_set.mini_batch(batch_size)
        _, loss_, prob_out = session.run([optimizer, loss, probability_output], feed_dict={model.x: mb_x, model.y: mb_y})
    else:
        sparse_index, sparse_ids, sparse_values, sparse_shape, mb_y = train_set.mini_batch(batch_size)
        _, loss_, prob_out = session.run([optimizer, loss, probability_output],
                                         feed_dict={model.sparse_index: sparse_index,
                                                    model.sparse_ids: sparse_ids,
                                                    model.sparse_values: sparse_values,
                                                    model.sparse_shape: sparse_shape,
                                                    model.y: mb_y})
    auc = roc_auc_score(mb_y, prob_out)
    print("epoch: ", train_set.epoch_pass, " auc: ", auc)