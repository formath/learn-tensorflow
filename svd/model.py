#!/usr/bin/env python

import tensorflow as tf

class SVDModel:
    def __init__(self, user_num, item_num, embedding_num):
        self.user_num = user_num
        self.item_num = item_num
        self.embedding_num = embedding_num

    def forward(self, batch_users, batch_items):
        '''
        Calucate preference for <user, item> pair
        :param batch_users: user batch. For example, [u19, u27, u23]
        :param batch_items: item batch. Must be the same size as batch_users. For example, [i37, i48, i21]
        :return: Tensor. the same size as batch_users and batch_items.
        For example, will return [<u19, i37>, <u27, i48>, <u23, i21>] where <u_id, i_id> indicates the u_id's preference for i_id
        '''
        with tf.device("/cpu:0"):
            user_embedding = tf.get_variable("user_embedding", shape=[self.user_num, self.embedding_num],
                                             initializer=tf.truncated_normal_initializer(stddev=0.02))
            item_embedding = tf.get_variable("item_embedding", shape=[self.item_num, self.embedding_num],
                                             initializer=tf.truncated_normal_initializer(stddev=0.02))
            user_bias = tf.get_variable("user_bias", shape=[self.user_num])
            item_bias = tf.get_variable("item_bias", shape=[self.item_num])
            batch_user_embedding = tf.nn.embedding_lookup(user_embedding, batch_users)
            batch_item_embedding = tf.nn.embedding_lookup(item_embedding, batch_items)
            batch_user_bias = tf.nn.embedding_lookup(user_bias, batch_users)
            batch_item_bias = tf.nn.embedding_lookup(item_bias, batch_items)
            return tf.add(tf.add(tf.reduce_sum(tf.mul(batch_user_embedding, batch_item_embedding), 1), batch_user_bias), batch_item_bias), batch_user_embedding, batch_item_embedding