#!/usr/bin/env python

import os
import tensorflow as tf

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            "userid": tf.FixedLenFeature([], tf.int64),
            "itemid": tf.FixedLenFeature([], tf.int64),
            "rating": tf.FixedLenFeature([], tf.float32)
        })
    userid = features["userid"]
    itemid = features["itemid"]
    rating = features["rating"]
    return userid, itemid, rating

def read_batch(file_name, max_epoch, batch_size, thread_num, min_after_dequeue):
    with tf.name_scope("input"):
        filename_queue = tf.train.string_input_producer(
            tf.train.match_filenames_once(file_name),
            num_epochs=max_epoch)
        userid, itemid, rating = read_and_decode(filename_queue)
        capacity = thread_num * batch_size + min_after_dequeue
        batch_userid, batch_itemid, batch_rating = tf.train.shuffle_batch(
            [userid, itemid, rating],
            batch_size=batch_size,
            num_threads=thread_num,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue)
        return batch_userid, batch_itemid, batch_rating