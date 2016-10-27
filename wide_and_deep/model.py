#!/usr/bin/env python

import tensorflow as tf

class WideAndDeepModel:
    def __init__(self, model_type, feature_num, label_num):
        self.model_type = model_type
        self.feature_num = feature_num
        self.label_num = label_num
        self.hidden1_num = 10
        self.hidden2_num = 10

    def forward(self, sparse_id, sparse_value):
        print("Model type: {} Feature num: {} Label num: {}".format(self.model_type, self.feature_num, self.label_num))
        if self.model_type == "wide":
            return self.wide_inference(sparse_id, sparse_value)
        elif self.model_type == "deep":
            return self.deep_inference(sparse_id, sparse_value)
        elif self.model_type == "wide_and_deep":
            return self.wide_and_deep_inference(sparse_id, sparse_value)
        else:
            print("Error: unknown model type")
            exit(1)

    def full_connect(self, input, weight_shape, bias_shape):
        with tf.device('/cpu:0'):
            weight = tf.get_variable("weight", weight_shape, initializer=tf.random_normal_initializer())
            bias = tf.get_variable("bias", bias_shape, initializer=tf.random_normal_initializer())
        return tf.matmul(input, weight) + bias

    def sparse_full_connect(self, sparse_id, sparse_value, weight_shape, bias_shape):
        with tf.device('/cpu:0'):
            weight = tf.get_variable("weight", weight_shape, initializer=tf.random_normal_initializer())
            bias = tf.get_variable("bias", bias_shape, initializer=tf.random_normal_initializer())
        return tf.nn.embedding_lookup_sparse(weight, sparse_id, sparse_value, combiner="sum") + bias

    def relu(self, input):
        return tf.nn.relu(input)

    def full_connect_relu(self, input, weight_shape, bias_shape):
        return tf.nn.relu(self.full_connect(input, weight_shape, bias_shape))

    def deep_inference(self, sparse_id, sparse_value):
        '''
        deep neural networks model
        :param sparse_id: sparse tensor holding sparse id list
        :param sparse_value: sparse tensor holding sparse value list that matches sparse id list
        :return: tensor
        '''
        with tf.variable_scope("dnn_layer1"):
            layer = self.sparse_full_connect(sparse_id, sparse_value, [self.feature_num, self.hidden1_num], [self.hidden1_num])
            layer = tf.nn.relu(layer)
        with tf.variable_scope("dnn_layer2"):
            layer = self.full_connect_relu(layer, [self.hidden1_num, self.hidden2_num], [self.hidden2_num])
        with tf.variable_scope("dnn_output"):
            layer = self.full_connect(layer, [self.hidden2_num, self.label_num], [self.label_num])
        return layer

    def wide_inference(self, sparse_id, sparse_value):
        '''
        sparse logistic regression model
        :param sparse_id: sparse tensor holding sparse id list
        :param sparse_values: sparse tensor holding sparse value list that matches sparse id list
        :return: tensor
        '''
        with tf.variable_scope("lr_output"):
            layer = self.sparse_full_connect(sparse_id, sparse_value, [self.feature_num, self.label_num], [self.label_num])
        return layer


    def wide_and_deep_inference(self, sparse_id, sparse_value):
        '''
        bagging of dnn and lr
        :param sparse_id: sparse tensor holding sparse id list
        :param sparse_value: sparse tensor holding sparse value list that matches sparse id list
        :return: tensor
        '''
        return self.wide_inference(sparse_id, sparse_value) + self.deep_inference(sparse_id, sparse_value)
