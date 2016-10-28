#!/usr/bin/env python

import os
import tensorflow as tf
from svd.model import SVDModel
from svd.read_data import read_batch

# config
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
flags.DEFINE_float('lam', 0.01, 'regularization for overcoming overfitting')
flags.DEFINE_integer('max_epoch', 100, ' max train epochs')
flags.DEFINE_integer("batch_size", 100, "batch size for sgd")
flags.DEFINE_integer("valid_batch_size", 100, "validate set batch size")
flags.DEFINE_integer("thread_num", 1, "number of thread to read data")
flags.DEFINE_integer("min_after_dequeue", 100, "min_after_dequeue for shuffle queue")
flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "checkpoint dirctory")
flags.DEFINE_string("tensorboard_dir", "./tensorboard/", "summary data saved for tensorboard")
flags.DEFINE_string("optimizer", "adagrad", "optimization algorithm")
flags.DEFINE_integer('steps_to_validate', 1, 'steps to validate and print')
flags.DEFINE_bool("train_from_checkpoint", False, "reload model from checkpoint and go on training")

if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
if not os.path.exists(FLAGS.tensorboard_dir):
    os.makedirs(FLAGS.tensorboard_dir)

# read train data
train_userid, train_itemid, train_rating = read_batch("../data/svd_data/train.data.tfrecord",
                                                FLAGS.max_epoch,
                                                FLAGS.batch_size,
                                                FLAGS.thread_num,
                                                FLAGS.min_after_dequeue)
# read validate data
valid_userid, valid_itemid, valid_rating = read_batch("../data/svd_data/test.data.tfrecord",
                                                FLAGS.max_epoch,
                                                FLAGS.batch_size,
                                                FLAGS.thread_num,
                                                FLAGS.min_after_dequeue)

# define model
max_userid = 671
max_itemid = 163949
embedding_num = 10
model = SVDModel(max_userid+1, max_itemid+1, embedding_num)

# define loss
preference, user_embedding, item_embedding = model.forward(train_userid, train_itemid)
rmse = tf.nn.l2_loss(tf.sub(preference, train_rating))
regularization = tf.add(tf.nn.l2_loss(user_embedding), tf.nn.l2_loss(item_embedding))
lam = tf.constant(FLAGS.lam, dtype=tf.float32, shape=[])
loss = rmse + tf.mul(regularization, lam)

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

# eval rmse
tf.get_variable_scope().reuse_variables()
valid_preference, _, _ = model.forward(valid_userid, valid_itemid)
valid_rmse = tf.nn.l2_loss(tf.sub(valid_preference, valid_rating))

# checkpoint
checkpoint_file = FLAGS.checkpoint_dir + "/checkpoint"
saver = tf.train.Saver()

# summary
tf.scalar_summary('loss', loss)
tf.scalar_summary('rmse', valid_rmse)
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
                valid_rmse_value, summary_value = sess.run([valid_rmse, summary_op])
                print("Step: {}, loss: {}, rmse: {}".format(
                    step, loss_value, valid_rmse_value))
                writer.add_summary(summary_value, step)
                saver.save(sess, checkpoint_file, global_step=step)
    except tf.errors.OutOfRangeError:
        print("training done")
    finally:
        coord.request_stop()

    # wait for threads to exit
    coord.join(threads)
    sess.close()