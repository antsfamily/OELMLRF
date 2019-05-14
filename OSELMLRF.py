#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-05-12 23:32:02
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
#
# @Note: Local Receptive Fields Based Online Sequential Extreme Learning Machine

import numpy as np
import tftool as tt
import tensorflow as tf


class OSELMLRF(object):

    r"""Class of OSELMLRF

    Local Receptive Fields Based Online Extreme Learning Machine
    """

    def __init__(self, sess, insize, outsize, name=None):
        r"""Class Initialization

        Initialize OSELMLRF class

        Arguments:
            sess {Session Object} -- tensorflow session

            layers {dict} -- ``{'input': [H, W, C],
                                'conv1': [kH, kW, kC, sH, sW, 'activation'],
                                'pool1': [kH, kW, kC, sH, sW, 'type'],
                                'conv2': [kH, kW, kC, sH, sW, 'activation'],
                                'pool2': [kH, kW, kC, sH, sW, 'type'],
                                .
                                .
                                .
                                'convn': [kH, kW, kC, sH, sW, 'activation'],
                                'pooln': [kH, kW, kC, sH, sW, 'type'],
                                'flatten': None,
                                'fc': [L, 'activation'] or None
                                'output': [m]
                                }``

        Keyword Arguments:
            name {string} -- OSELMLRF Object name (default: {None})
        """

        self._name = name
        self._sess = sess
        self._insize = insize  # [H, W, C]
        self._L = -1  # number of H
        self._outsize = outsize  # [m]
        self._X = -1  # input node
        self._T = -1  # output node
        self._Y = -1  # output node
        self._C = -1  # weight factor
        self._H = -1  # output features of the last hidden layer
        self._P = -1
        self._W = {}  # hidden layer weights
        self._B = -1  # output weight Beta
        self._net = {}

    def build(self, dtype=tf.float32, randseed=None):
        """"

        build OSELMLRF net

        Keyword Arguments:
            dtype {tensorflow type} -- type of net (default: {tf.float32})
            randseed {integer} -- seed for generating random numbers (default: {None})
        """

        # generates the same repeatable sequence
        tf.set_random_seed(randseed)

        self._X = tf.placeholder(
            dtype=dtype, shape=[None] + self._insize, name='input')
        self._T = tf.placeholder(
            dtype=dtype, shape=[None] + self._outsize, name='output')
        self._C = tf.placeholder(dtype=tf.float32, shape=1, name='factor')

        self._net['X'] = self._X
        self._net['Y'] = self._Y
        self._net['T'] = self._T

        InitA = tf.orthogonal_initializer(gain=1.0, dtype=dtype)

        # ----------------------------gen weights

        self._W['C1'] = tf.get_variable(
            name='C1', shape=[3, 3, self._insize[2], 8],
            dtype=dtype, initializer=InitA, trainable=False)

        self._W['C2'] = tf.get_variable(
            name='C2', shape=[3, 3, 8, 16],
            dtype=dtype, initializer=InitA, trainable=False)

        self._W['C3'] = tf.get_variable(
            name='C3', shape=[3, 3, 16, 32],
            dtype=dtype, initializer=InitA, trainable=False)

        # ----------------------------gen net
        self._net['C1'] = tf.nn.conv2d(
            self._X, self._W['C1'],
            strides=[1, 1, 1, 1],  # [1, sH, sW, 1]
            padding="VALID", data_format='NHWC',
            dilations=[1, 1, 1, 1], name='C1')

        # self._net['P1'] = tf.nn.max_pool(
        self._net['P1'] = tt.nn.square_root_pool(
            self._net['C1'],
            ksize=[1, 2, 2, 1],  # [1, kH, kW, 1]
            strides=[1, 1, 1, 1],  # [1, sH, sW, 1]
            padding="VALID", data_format='NHWC', name='P1')

        self._net['C2'] = tf.nn.conv2d(
            self._net['P1'], self._W['C2'],
            strides=[1, 1, 1, 1],  # [1, sH, sW, 1]
            padding="VALID", data_format='NHWC',
            dilations=[1, 1, 1, 1], name='C2')

        # self._net['P2'] = tf.nn.max_pool(
        self._net['P2'] = tt.nn.square_root_pool(
            self._net['C2'],
            ksize=[1, 2, 2, 1],  # [1, kH, kW, 1]
            strides=[1, 1, 1, 1],  # [1, sH, sW, 1]
            padding="VALID", data_format='NHWC', name='P2')

        self._net['C3'] = tf.nn.conv2d(
            self._net['P2'], self._W['C3'],
            strides=[1, 1, 1, 1],  # [1, sH, sW, 1]
            padding="VALID", data_format='NHWC',
            dilations=[1, 1, 1, 1], name='C3')

        # self._net['P3'] = tf.nn.max_pool(
        self._net['P3'] = tt.nn.square_root_pool(
            self._net['C3'],
            ksize=[1, 2, 2, 1],  # [1, kH, kW, 1]
            strides=[1, 1, 1, 1],  # [1, sH, sW, 1]
            padding="VALID", data_format='NHWC', name='P3')

        HO = self._net['P3']
        _, H, W, C = HO.get_shape()
        self._L = H * W * C

        # H --> N - H * W * C
        self._H = tf.reshape(HO, shape=[-1, self._L], name='H')

        print("X --> ", self._X.get_shape())
        print("T --> ", self._T.get_shape())
        print("C --> ", self._C.get_shape())
        print("C --> ", HO.get_shape())
        print("H --> ", self._H.get_shape())

        print(self._L, self._outsize)
        self._B = tf.get_variable(
            name='B', shape=[self._L] + self._outsize,
            dtype=dtype, trainable=False)

    def train(self, X, T):
        pass
