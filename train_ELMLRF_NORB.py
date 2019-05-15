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

nC = 5  # 1,2,3,4,5
H, W, C = [32, 32, 2]

traindatafile = '/mnt/d/DataSets/oi/nsi/NORB/norb_traindata.mat'
data = scio.loadmat(traindatafile, struct_as_record=True)

Xtrain = data['X']  # 2048-24300
Ttrain = data['Y'] - 1  # 24300-1

Ttrain = Ttrain.flatten()  # 24300
print(Ttrain[0:10])

Xtrain = np.transpose(Xtrain)  # 24300-2048
Ttrain = np.eye(nC)[Ttrain]  # 24300-5

Y = np.argmax(Ttrain, axis=1)
print(Y[0:10])


print(Xtrain.shape, Ttrain.shape)

testdatafile = '/mnt/d/DataSets/oi/nsi/NORB/norb_testdata.mat'
data = scio.loadmat(testdatafile, struct_as_record=True)

Xval = data['X']  # 2048-24300
Tval = data['Y'] - 1  # 24300-1
Tval = Tval.flatten()  # 24300
print(Tval[0:10])

Xval = np.transpose(Xval)  # 24300-2048
Tval = np.eye(nC)[Tval]  # 24300-5

Y = np.argmax(Tval, axis=1)
print(Y[0:10])

Ns = 10000
Ns = 24300

Xtrain = Xtrain[0:Ns, :]
Ttrain = Ttrain[0:Ns, :]
Xval = Xval[0:24300, :]
Tval = Tval[0:24300, :]

print(np.min(Xtrain), np.max(Xtrain))
print(np.min(Xval), np.max(Xval))

N = np.size(Xtrain, 0)
Xtrain = np.reshape(Xtrain, [N, H, W, C])

N = np.size(Xval, 0)
Xval = np.reshape(Xval, [N, H, W, C])

print(Xtrain.shape, Xval.shape)

insize = [H, W, C]
outsize = [nC]

Cs = [100.0, 10.0, 5.0, 2.0, 1.5, 0.01, 0.1,
      0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

randseed = 0
dtype = tf.float32

sess = tf.Session()

elmlrfnet = elmlrf.ELMLRF(sess, insize, outsize, name='elmlrf')

elmlrfnet.build(dtype=dtype, randseed=randseed)

elmlrfnet.trainval(Xtrain, Ttrain, Xval, Tval, Cs)
