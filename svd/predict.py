#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from svd.model import SVDModel

# config
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "checkpoint dirctory")

# define model
max_userid = 671
max_itemid = 163949
embedding_num = 10
model = SVDModel(max_userid+1, max_itemid+1, embedding_num)

# predict op
userid = tf.placeholder(tf.int64)
itemid = tf.placeholder(tf.int64)
preference, _, _ = model.forward(userid, itemid)

# checkpoint
saver = tf.train.Saver()

# session run
with tf.Session() as sess:
    test_file = "../data/svd_data/test.data"
    output_file = "../data/svd_data/test.output"
    userids = []
    itemids = []
    for line in open(test_file, "r"):
        uid, iid, _, _ = line.split(",")
        userids.append(int(uid))
        itemids.append(int(iid))

    checkpoint_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if checkpoint_state and checkpoint_state.model_checkpoint_path:
        print("Checkpoint {}".format(checkpoint_state.model_checkpoint_path))
        saver.restore(sess, checkpoint_state.model_checkpoint_path)
        output = sess.run(
            preference,
            feed_dict={userid: userids,
                       itemid: itemids})
        np.savetxt(output_file, output, delimiter=",")
        print("Save result to file: {}".format(output_file))
    else:
        print("Error: no checkpoint found")
        exit(1)