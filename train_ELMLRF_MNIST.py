#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-05-12 23:32:02
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
#
# @Note: Local Receptive Fields Based Online Sequential Extreme Learning Machine

import os
import elmlrf
import numpy as np
import tensorflow as tf
import scipy.io as scio

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

nC = 10
H, W, C = [28, 28, 1]

datafile = './data/MNIST/mnist_uint8.mat'
data = scio.loadmat(datafile, struct_as_record=True)

Xtrain = data['train_x']  # 60000-784
Ttrain = data['train_y']  # 60000-10
Xval = data['test_x']  # 10000-784
Tval = data['test_y']  # 10000-10

Xtrain = Xtrain[1:1000, :]
Ttrain = Ttrain[1:1000, :]
Xval = Xval[1:1000, :]
Tval = Tval[1:1000, :]

N = np.size(Xtrain, 0)
Xtrain = np.reshape(Xtrain, [N, H, W, C])

N = np.size(Xval, 0)
Xval = np.reshape(Xval, [N, H, W, C])

print(Xtrain.shape, Xval.shape)

insize = [H, W, C]
outsize = [nC]
Cs = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

randseed = 2019
dtype = tf.float32

sess = tf.Session()

elmlrfnet = elmlrf.ELMLRF(sess, insize, outsize, name='elmlrf')

elmlrfnet.build(dtype=dtype, randseed=randseed)

elmlrfnet.trainval(Xtrain, Ttrain, Xval, Tval, Cs)
