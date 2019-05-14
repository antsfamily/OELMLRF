#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-05-12 23:32:02
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
#
# @Note: Local Receptive Fields Based Online Sequential Extreme Learning Machine

import os
import oselmlrf
import tensorflow as tf


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


H, W, C = (32, 32, 2)
nC = 5


insize = [H, W, C]
outsize = [nC]

randseed = 2019
dtype = tf.float32

sess = tf.Session()

oselmlrfnet = oselmlrf.OSELMLRF(sess, insize, outsize, name='oselmlrf')

oselmlrfnet.build(dtype=dtype, randseed=randseed)
