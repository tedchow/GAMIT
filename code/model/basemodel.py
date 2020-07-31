from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.layers import base

class BaseModel(base.Layer):
    def __init__(self, config):
        self.phs = {}  # placeholders
        self.config = config
        # self.mode = mode
        # self.lambda_l2_reg = tf.constant(hps.lambda_l2_reg, dtype=tf.float32)

        with tf.variable_scope('inputs'):
            # the input of global spatial attention, [batch_size, n_steps_encoder, n_links]
            self.phs['inputs'] = tf.placeholder(tf.float32,
                                                       [None, config.len_his] + [config.n_links, config.len_his2] + [config.len_f], # None denotes the batchsize
                                                       name='inputs')
           
        with tf.variable_scope('groundtruth'):
            # Ground truth, [batch_size, n_steps_decoder, n_output_decoder], if no multi-task, n_output_decoder = 1
            self.phs['labels'] = tf.placeholder(tf.float32,
                                                [None, config.len_pre] + [config.n_links, config.len_his2] + [config.len_f],
                                                name='labels')
        self.phs['preds'] = tf.placeholder(tf.float32,
                                                [None, config.len_pre] + [config.n_links, config.len_his2] + [config.len_f],
                                                name='preds')


        self.phs['scale_labels'] = tf.placeholder(tf.float32,
                                                [None, config.len_pre] + [config.n_links, config.len_his2] + [config.len_f],
                                                name='scale_labels')  
        #     self.phs['labels'] = tf.placeholder(tf.float32,
        #                                         [None, config.len_pre, config.n_links, config.len_pre] + [1],
        #                                         name='labels')
        # self.phs['preds'] = tf.placeholder(tf.float32,
        #                                         [None, config.len_pre, config.n_links, config.len_pre] + [1],
        #                                         name='preds')

    def build(self):
        pass

    @property
    def get_metric(self):
        metric_list = [root_mean_squared_error(self.phs['labels'], self.phs['preds']),
                       mean_absolute_error(self.phs['labels'], self.phs['preds'])]
        return metric_list

    def get_loss(self):
        pass

    def get_l2reg_loss(self):
        pass

    @property
    def loss(self):
        with tf.variable_scope('Loss'):
            return self.get_loss() + self.get_l2reg_loss()

    @property
    def train_op(self):
        pass

    def init(self, sess):
        sess.run(tf.global_variables_initializer())

    def summary(self, hps):
        pass
        
    def mod_fn(self):
        pass
