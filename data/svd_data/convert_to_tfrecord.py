#!/usr/bin/env python

import tensorflow as tf
import os

# data schema: userid,itemid,rating
# 1,31,2.5,1260759144
# 1,1029,3.0,1260759179
# 1,1061,3.0,1260759182
# 1,1129,2.0,1260759185
# 1,1172,4.0,1260759205

def convert_tfrecords(input_filename, output_filename):
    current_path = os.getcwd()
    input_file = os.path.join(current_path, input_filename)
    output_file = os.path.join(current_path, output_filename)
    print("Start to convert {} to {}".format(input_file, output_file))

    writer = tf.python_io.TFRecordWriter(output_file)

    for line in open(input_file, "r"):
        userid, itemid, rating, _ = line.split(",")

        # Write each example one by one
        example = tf.train.Example(features=tf.train.Features(feature={
            "userid":
                tf.train.Feature(int64_list=tf.train.Int64List(value=[int(userid)])),
            "itemid":
                tf.train.Feature(int64_list=tf.train.Int64List(value=[int(itemid)])),
            "rating":
                tf.train.Feature(float_list=tf.train.FloatList(value=[float(rating)]))
        }))

        writer.write(example.SerializeToString())

    writer.close()
    print("Successfully convert {} to {}".format(input_file, output_file))


current_path = os.getcwd()
for file in os.listdir(current_path):
    if file.endswith(".data"):
        convert_tfrecords(file, file + ".tfrecord")