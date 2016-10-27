#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from wide_and_deep.model import WideAndDeepModel

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "checkpoint dirctory")
flags.DEFINE_string("model_type", "wide_and_deep", "model type, option: wide, deep, wide_and_deep")

# define model
feature_num = 124
label_num = 2
model = WideAndDeepModel(FLAGS.model_type, feature_num, label_num)

# predict op
sparse_index = tf.placeholder(tf.int64)
sparse_id = tf.placeholder(tf.int64)
sparse_value = tf.placeholder(tf.float32)
sparse_shape = tf.placeholder(tf.int64)
test_id = tf.SparseTensor(sparse_index, sparse_id, sparse_shape)
test_value = tf.SparseTensor(sparse_index, sparse_value, sparse_shape)
logits = model.forward(test_id, test_value)
softmax = tf.nn.softmax(logits)
predict_op = tf.argmax(softmax, 1)

# checkpoint
saver = tf.train.Saver()

# session run
with tf.Session() as sess:
    test_file = "../data/libsvm_data/test.data"
    output_file = "../data/libsvm_data/test.output"
    feature_ids = []
    feature_values = []
    feature_index = []
    ins_num = 0
    for line in open(test_file, "r"):
        tokens = line.split(" ")
        feature_num = 0
        for feature in tokens[1:]:
            feature_id, feature_value = feature.split(":")
            feature_ids.append(int(feature_id))
            feature_values.append(float(feature_value))
            feature_index.append([ins_num, feature_num])
            feature_num += 1
        ins_num += 1

    checkpoint_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if checkpoint_state and checkpoint_state.model_checkpoint_path:
        print("Checkpoint {}".format(checkpoint_state.model_checkpoint_path))
        saver.restore(sess, checkpoint_state.model_checkpoint_path)
        output = sess.run(
                softmax,
                predict_op,
                feed_dict={sparse_index: feature_index,
                           sparse_id: feature_ids,
                           sparse_value: feature_value,
                           sparse_shape: [ins_num, feature_num]})
        np.savetxt(output_file, output, delimiter=",")
        print("Save result to file: {}".format(output_file))
    else:
        print("Error: no checkpoint found")
        exit(1)