#!/usr/bin/env python

import sys
import os
import numpy as np
import tensorflow as tf

class DataSet(object):
    def __init__(self):
        self.iter = 0
        self.epoch_pass = 0

    def load(self, file, feature_num, dense = False):
        self.feature_num = feature_num
        self.ins_num = 0
        f = open(file, "r")
        if dense:
            self.data_type = "dense"
            self.x = []
            self.y = []
            for line in f.readlines():
                tokens = line.split(" ")
                if len(tokens) != self.feature_num + 1:
                    exit("dense feature num error")
                self.y.append(float(tokens[0]))
                self.x.append([float(value) for value in tokens[1:]])
                self.ins_num += 1
        else:
            self.data_type = "sparse"
            self.y = []
            self.feature_ids = []
            self.feature_values = []
            self.ins_feature_interval = []
            self.ins_feature_interval.append(0)
            for line in f.readlines():
                tokens = line.split(" ")
                self.y.append(float(tokens[0]))
                self.ins_feature_interval.append(self.ins_feature_interval[-1] + len(tokens) - 1)
                for feature in tokens[1:]:
                    feature_id, feature_value = feature.split(":")
                    self.feature_ids.append(int(feature_id))
                    self.feature_values.append(float(feature_value))
                self.ins_num += 1

    def mini_batch(self, batch_size):
        begin = self.iter
        end = self.iter
        if self.iter + batch_size > self.ins_num:
            end = self.ins_num
            self.iter = 0
            self.epoch_pass += 1
        else:
            end += batch_size
            self.iter = end
        return self.slice(begin, end)

    def slice(self, begin, end):
        if self.data_type == "dense":
            x = np.array(self.x[begin:end])
            y = np.array(self.y[begin:end]).reshape((end - begin, 1))
            return (x, y)
        else:
            sparse_index = []
            sparse_ids = []
            sparse_values = []
            sparse_shape = []
            max_feature_num = 0
            for i in range(begin, end):
                feature_num = self.ins_feature_interval[i + 1] - self.ins_feature_interval[i]
                if feature_num > max_feature_num:
                    max_feature_num = feature_num
                for j in range(self.ins_feature_interval[i], self.ins_feature_interval[i + 1]):
                    sparse_index.append([i - begin, j - self.ins_feature_interval[i]]) # index must be accent
                    sparse_ids.append(self.feature_ids[j])
                    sparse_values.append(self.feature_values[j])
            sparse_shape.append(end - begin)
            sparse_shape.append(max_feature_num)
            y = np.array(self.y[begin:end]).reshape((end - begin, 1))
            return (sparse_index, sparse_ids, sparse_values, sparse_shape, y)