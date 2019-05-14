#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-05-12 23:32:02
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
#
# @Note: Local Receptive Fields Based Online Sequential Hierarchical Extreme Learning Machine

import numpy as np
import tftool as tt
import tensorflow as tf


class OSHELMLRF(object):

    r"""Class of OSELMLRF

    Local Receptive Fields Based Online Extreme Learning Machine
    """

    def __init__(self, sess, layers, name=None):
        r"""Class Initialization

        Initialize OSELMLRF class

        Arguments:
            sess {Session Object} -- tensorflow session

            layers {dict} -- ``{'input': [H, W, C],
                                'conv1': ['conv2d', [kH, kW, kC, sH, sW], 'activation'],
                                'pool1': ['square_root_pool', [kH, kW, kC, sH, sW], 'type'],
                                'conv2': ['conv2d', [kH, kW, kC, sH, sW], 'activation'],
                                'pool2': ['square_root_pool', [kH, kW, kC, sH, sW], 'type'],
                                .
                                .
                                .
                                'convn': ['conv2d', [kH, kW, kC, sH, sW], 'activation'],
                                'pooln': ['square_root_pool', [kH, kW, kC, sH, sW], 'type'],
                                'flatten': None,
                                'fc': [L, 'activation'] or None
                                'output': [m]
                                }``

        Keyword Arguments:
            name {string} -- OSELMLRF Object name (default: {None})
        """

        self._name = name
        self._sess = sess
        self._layers = layers
        self._L = -1  # number of H
        self._m = layers['output']
        self._X = -1  # input node
        self._T = -1  # output node
        self._C = -1  # weight factor
        self._H = -1  # output features of the last hidden layer
        self._P = -1
        self._W = -1  # hidden layer weights
        self._B = -1  # output weight Beta
        self._hnet = {}

    def build(self, dtype=tf.float32, randseed=None):
        """build net

        build OSELMLRF net
        """

        tf.set_random_seed(randseed)

        self._X = tf.placeholder(dtype, [None] + self._layers['input'], 'input')
        self._T = tf.placeholder(dtype, [None] + self._layers['output'], 'output')
        self._C = tf.placeholder(dtype, [1], 'weight factor')

        for layername, layerparam in self._layers.item():
            if layername[-4] 
            layers['layername']
            eval('tt.nn.' + layerparam[0])()
            tf.nn.max_pool
            self._hnet[layername] = tt.nn.square_root_pool


