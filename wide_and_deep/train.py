#!/usr/bin/env python

import os
import datetime
import tensorflow as tf
from wide_and_deep.model import WideAndDeepModel
from wide_and_deep.read_data import read_batch

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
flags.DEFINE_integer('max_epoch', 100, ' max train epochs')
flags.DEFINE_integer("batch_size", 100, "batch size for sgd")
flags.DEFINE_integer("valid_batch_size", 100, "validate set batch size")
flags.DEFINE_integer("thread_num", 1, "number of thread to read data")
flags.DEFINE_integer("min_after_dequeue", 100, "min_after_dequeue for shuffle queue")
flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "checkpoint dirctory")
flags.DEFINE_string("tensorboard_dir", "./tensorboard/", "summary data saved for tensorboard")
flags.DEFINE_string("model_type", "wide_and_deep", "model type, option: wide, deep, wide_and_deep")
flags.DEFINE_string("optimizer", "adagrad", "optimization algorithm")
flags.DEFINE_integer('steps_to_validate', 10, 'steps to validate and print')
flags.DEFINE_bool("train_from_checkpoint", False, "reload model from checkpoint and go on training")

if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
if not os.path.exists(FLAGS.tensorboard_dir):
    os.makedirs(FLAGS.tensorboard_dir)

# read train data
train_label, train_id, train_value = read_batch("../data/libsvm_data/train.data.tfrecord",
                                                FLAGS.max_epoch,
                                                FLAGS.batch_size,
                                                FLAGS.thread_num,
                                                FLAGS.min_after_dequeue)
# read validate data
valid_label, valid_id, valid_value = read_batch("../data/libsvm_data/test.data.tfrecord",
                                                FLAGS.max_epoch,
                                                FLAGS.batch_size,
                                                FLAGS.thread_num,
                                                FLAGS.min_after_dequeue)

# define model
feature_num = 124
label_num = 2
model = WideAndDeepModel(FLAGS.model_type, feature_num, label_num)

# define loss
logits = model.forward(train_id, train_value)
train_label = tf.to_int64(train_label)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, train_label)
loss = tf.reduce_mean(cross_entropy, name='loss')

# define optimizer
print("Optimization algorithm: {}".format(FLAGS.optimizer))
if FLAGS.optimizer == "sgd":
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
elif FLAGS.optimizer == "adadelta":
    optimizer = tf.train.AdadeltaOptimizer(FLAGS.learning_rate)
elif FLAGS.optimizer == "adagrad":
    optimizer = tf.train.AdagradOptimizer(FLAGS.learning_rate)
elif FLAGS.optimizer == "adam":
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
elif FLAGS.optimizer == "ftrl":
    optimizer = tf.train.FtrlOptimizer(FLAGS.learning_rate)
elif FLAGS.optimizer == "rmsprop":
    optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate)
else:
    print("Error: unknown optimizer: {}".format(FLAGS.optimizer))
    exit(1)
with tf.device("/cpu:0"):
    global_step = tf.Variable(0, name='global_step', trainable=False)
train_op = optimizer.minimize(loss, global_step=global_step)

# eval acc
tf.get_variable_scope().reuse_variables()
valid_logits = model.forward(valid_id, valid_value)
valid_softmax = tf.nn.softmax(valid_logits)
valid_label = tf.to_int64(valid_label)
correct_prediction = tf.equal(tf.argmax(valid_softmax, 1), valid_label)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# eval auc
valid_label = tf.cast(valid_label, tf.int32)
sparse_labels = tf.reshape(valid_label, [-1, 1])
derived_size = tf.shape(valid_label)[0]
indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
concated = tf.concat(1, [indices, sparse_labels])
outshape = tf.pack([derived_size, label_num])
new_valid_label = tf.sparse_to_dense(concated, outshape, 1.0, 0.0)
_, auc_op = tf.contrib.metrics.streaming_auc(valid_softmax, new_valid_label)

# checkpoint
checkpoint_file = FLAGS.checkpoint_dir + "/checkpoint"
saver = tf.train.Saver()

# summary
tf.scalar_summary('loss', loss)
tf.scalar_summary('accuracy', accuracy)
tf.scalar_summary('auc', auc_op)
summary_op = tf.merge_all_summaries()

# train loop
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    writer = tf.train.SummaryWriter(FLAGS.tensorboard_dir, sess.graph)
    sess.run(init_op)
    sess.run(tf.initialize_local_variables())

    if FLAGS.train_from_checkpoint:
        checkpoint_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if checkpoint_state and checkpoint_state.model_checkpoint_path:
            print("Continue training from checkpoint {}".format(checkpoint_state.model_checkpoint_path))
            saver.restore(sess, checkpoint_state.model_checkpoint_path)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    try:
        while not coord.should_stop():
            _, loss_value, step = sess.run([train_op, loss, global_step])
            if step % FLAGS.steps_to_validate == 0:
                accuracy_value, auc_value, summary_value = sess.run([accuracy, auc_op, summary_op])
                print("Step: {}, loss: {}, accuracy: {}, auc: {}".format(
                        step, loss_value, accuracy_value, auc_value))
                writer.add_summary(summary_value, step)
                saver.save(sess, checkpoint_file, global_step=step)
    except tf.errors.OutOfRangeError:
        print("training done")
    finally:
        coord.request_stop()

    # wait for threads to exit
    coord.join(threads)
    sess.close()