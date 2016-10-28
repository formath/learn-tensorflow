#!/usr/bin/env python

import tensorflow as tf
import os

# Read TFRecords file
current_path = os.getcwd()
tfrecords_file_name = "train.data.tfrecord"
input_file = os.path.join(current_path, tfrecords_file_name)

# Constrain the data to print
max_print_number = 100
print_number = 1

for serialized_example in tf.python_io.tf_record_iterator(input_file):
    # Get serialized example from file
    example = tf.train.Example()
    example.ParseFromString(serialized_example)

    # Read data in specified format
    userid = example.features.feature["userid"].int64_list.value
    itemid = example.features.feature["itemid"].int64_list.value
    rating = example.features.feature["rating"].float_list.value
    print("Number: {}, userid: {}, itemid: {}, rating: {}".format(print_number,
                                                                  userid, itemid, rating))

    # Return when reaching max print number
    if print_number > max_print_number:
        exit()
    else:
        print_number += 1