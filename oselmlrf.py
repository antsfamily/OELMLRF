#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-05-12 23:32:02
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
#
# @Note: Local Receptive Fields based Online Sequential Extreme Learning Machine

import pytool
import numpy as np
import tftool as tt
import tensorflow as tf


class OSELMLRF(object):

    r"""Class of OSELMLRF

    Local Receptive Fields based Online Sequential Extreme Learning Machine
    """

    def __init__(self, sess, insize, outsize, name=None):
        r"""Class Initialization

        Initialize OSELMLRF class

        Arguments:
            sess {Session Object} -- tensorflow session
                insize {list or tuple} -- net in size
                outsize {list or tuple} -- net out size

        Keyword Arguments:
            name {string} -- OSELMLRF Object name (default: {None})
        """

        self._name = name
        self._sess = sess
        self._insize = insize  # [H, W, C]
        self._outsize = outsize  # [m]
        # self._X = -1  # input
        # self._T = -1  # targets
        # self._Y = -1  # predict
        self._C = -1  # weight factor
        self._H = -1  # output features of the last hidden layer
        self._W = {}  # hidden layer weights
        self._P = -1  # output weight Beta
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

        print("===Build ELMLRF...")
        print("---Set random seed: " + str(randseed))
        tf.set_random_seed(randseed)

        print("---Graph node...")
        self._net['X'] = tf.placeholder(
            dtype=dtype, shape=[None] + self._insize, name='inputs')
        self._net['T'] = tf.placeholder(
            dtype=dtype, shape=[None] + self._outsize, name='targets')
        self._net['KP'] = tf.placeholder(
            dtype=dtype, name='keep_prob')

        # ----------------------------gen weights
        print("---Generates ELMLRF random weights(orthogonal)...")
        InitA = tf.orthogonal_initializer(gain=1.0, dtype=dtype)
        # InitA = tf.ones_initializer(dtype=dtype)
        self._W['C1'] = tf.get_variable(
            name='C1', shape=[4, 4, self._insize[2], 32],
            dtype=dtype, initializer=InitA, trainable=False)

        self._W['C2'] = tf.get_variable(
            name='C2', shape=[3, 3, 32, 32],
            dtype=dtype, initializer=InitA, trainable=False)

        self._W['C3'] = tf.get_variable(
            name='C3', shape=[3, 3, 32, 32],
            dtype=dtype, initializer=InitA, trainable=False)

        # ----------------------------gen net
        print("---Construct ELMLRF Net...")
        self._net['C1'] = tf.nn.conv2d(
            self._net['X'], self._W['C1'],
            strides=[1, 1, 1, 1],  # [1, sH, sW, 1]
            padding="VALID", data_format='NHWC',
            dilations=[1, 1, 1, 1], name='C1')

        self._net['C1'] = tf.nn.swish(self._net['C1'])

        # self._net['P1'] = tf.nn.max_pool(
        self._net['P1'] = tt.nn.square_root_pool(
            self._net['C1'],
            ksize=[1, 3, 3, 1],  # [1, kH, kW, 1]
            strides=[1, 1, 1, 1],  # [1, sH, sW, 1]
            padding="VALID", data_format='NHWC', name='P1')

        # self._net['P1'] = tf.nn.lrn(self._net['P1'])
        self._net['C2'] = tf.nn.conv2d(
            self._net['P1'], self._W['C2'],
            strides=[1, 1, 1, 1],  # [1, sH, sW, 1]
            padding="VALID", data_format='NHWC',
            dilations=[1, 1, 1, 1], name='C2')

        self._net['C2'] = tf.nn.swish(self._net['C2'])

        # self._net['P2'] = tf.nn.max_pool(
        self._net['P2'] = tt.nn.square_root_pool(
            self._net['C2'],
            ksize=[1, 3, 3, 1],  # [1, kH, kW, 1]
            strides=[1, 1, 1, 1],  # [1, sH, sW, 1]
            padding="VALID", data_format='NHWC', name='P2')

        # self._net['P2'] = tf.nn.lrn(self._net['P2'])

        self._net['C3'] = tf.nn.conv2d(
            self._net['P2'], self._W['C3'],
            strides=[1, 1, 1, 1],  # [1, sH, sW, 1]
            padding="VALID", data_format='NHWC',
            dilations=[1, 1, 1, 1], name='C3')

        self._net['C3'] = tf.nn.swish(self._net['C3'])

        # self._net['P3'] = tf.nn.max_pool(
        self._net['P3'] = tt.nn.square_root_pool(
            self._net['C3'],
            ksize=[1, 3, 3, 1],  # [1, kH, kW, 1]
            strides=[1, 1, 1, 1],  # [1, sH, sW, 1]
            padding="VALID", data_format='NHWC', name='P3')

        # self._net['P3'] = tf.nn.lrn(self._net['P3'])

        # HO = self._net['P3']
        HO = self._net['P1']
        # HO = tf.nn.swish(self._net['P1'])
        # HO = tf.nn.lrn(HO)
        # HO = tf.nn.swish(HO)
        # HO = tf.nn.swish(HO)
        # HO = self._net['C1']
        HO = tf.nn.dropout(HO, self._net['KP'])
        N, H, W, C = HO.get_shape()

        # H --> N - H * W * C
        self._net['H'] = tf.reshape(HO, shape=[-1, H * W * C], name='H')

        print("~~~X --> ", self._net['X'].get_shape())
        print("~~~T --> ", self._net['T'].get_shape())
        print("~~~H --> ", self._net['H'].get_shape())
        print("~~~HO --> ", HO.get_shape())

        print("---Initialize variables...")
        varlist = []
        for k, v in self._W.items():
            print(k, v)
            varlist.append(v)
        # varlist.append(self._B)
        self._sess.run(tf.initialize_variables(varlist))
        print("===Done!")

    def _computeB(self, H, T, C):
        """compute output weights

        [description]
        """
        if H is -1:
            raise ValueError("You need to train ELMLRF firstly!")

        self._C = C
        N, L = H.shape

        print("---N-L:", N, L)
        assert N > L

        self._P = np.linalg.inv(
            np.eye(L, L) / self._C + np.matmul(H.transpose(), H))

        self._B = np.matmul(np.matmul(self._P, H.transpose()), T)

    def _updateBcbyc(self, H, T):
        """update output weight chunk by chunk

        [description]

        Arguments:
            H {[type]} -- [description]
            T {[type]} -- [description]
        """

        N, _ = H.shape
        if self._P is -1:
            raise ValueError("Run self._computeB() firstly!")

        self._P = self._P - np.matmul(np.matmul(np.matmul(self._P, H.transpose()),
                                                np.linalg.inv(np.eye(N, N) + np.matmul(np.matmul(H, self._P), H.transpose()))),
                                      np.matmul(H, self._P))

        self._B = self._B + np.matmul(np.matmul(self._P, H.transpose()),
                                      (T - np.matmul(H, self._B)))

    def _updateBobyo(self, h, t, C):
        """update output weight one by one

        [description]

        Arguments:
            h {[type]} -- [description]
            t {[type]} -- [description]
            C {[type]} -- [description]
        """

        self._P = self._P - np.matmul(np.matmul(self._P, np.matmul(h, h.transpose())), self._P) / \
            (1 + np.matmul(np.matmul(h.transpose(), self._P), h))

        self._B = self._B + np.matmul(np.matmul(self._P, h),
                                      (t - np.matmul(h.transpose(), self._B)))

    def _predictY(self, H=-1):
        """predict

        Y = HB

        Returns:
                [type] -- [description]
        """
        if self._B is -1 or H is -1:
            raise ValueError("You need to train ELMLRF firstly!")
        return np.matmul(H, self._B)

    def _accuracy(self, Y, T):
        """compute accuracy

        compute accuracy

        Arguments:
            T {ndarray} -- Targets

        Returns:
            float number -- accuracy
        """

        correct_prediction = np.equal(
            np.argmax(Y, axis=1), np.argmax(T, axis=1))
        acc = np.mean(correct_prediction)
        return acc

    def ostrainval(self, Xtrain, Ttrain, Xval, Tval, Cs, batch_size, epochs):

        print("===In Training Validing")
        print("---Start Training...")

        print("---Forward propagation...")

        nTrain = np.size(Xtrain, 0)
        nVal = np.size(Xval, 0)

        Htrain = self._sess.run(
            self._net['H'], {self._net['X']: Xtrain, self._net['T']: Ttrain, self._net['KP']: 1.0})

        Hval = self._sess.run(
            self._net['H'], {self._net['X']: Xval, self._net['T']: Tval, self._net['KP']: 1.0})

        for C in Cs:
            print("---")
            print("---Compute beta...")
            print("~~~balance factor: ", C)
            self._computeB(Htrain, Ttrain, C)

            print("---Prediction...")

            Y = self._predictY(Htrain)
            acc = self._accuracy(Y, Ttrain)
            print("~~~accuracy: %.4f" % (acc), " <--train<--ELMLRF")

            Y = self._predictY(Hval)
            acc = self._accuracy(Y, Tval)
            print("~~~accuracy: %.4f" % (acc), " <--valid<--ELMLRF")

            print("---Online learning...")
            for epoch in range(epochs):

                total_batch = int(nTrain / batch_size)
                for batch in pytool.get_batches(Xtrain, Ttrain, batch_size, shuffle=True):
                    Xbatch, Tbatch = batch
                    Hbatch = self._sess.run(
                        self._net['H'], {self._net['X']: Xbatch, self._net['T']: Tbatch,
                                         self._net['KP']: 1.0})
                    self._updateBcbyc(Hbatch, Tbatch)

                print("---Prediction, epoch: ", epoch)

                Y = self._predictY(Htrain)
                acc = self._accuracy(Y, Ttrain)
                print("~~~accuracy: %.4f" % (acc), " <--train<--ELMLRF")

                Y = self._predictY(Hval)
                acc = self._accuracy(Y, Tval)
                print("~~~accuracy: %.4f" % (acc), " <--valid<--ELMLRF")

        print("---Done!")

    def trainval(self, Xtrain, Ttrain, Xval, Tval, Cs):

        print("===In Training Validing")
        print("---Start Training...")

        print("---Forward propagation...")
        Htrain = self._sess.run(
            self._net['H'], {self._net['X']: Xtrain, self._net['T']: Ttrain, self._net['KP']: 1.0})

        Hval = self._sess.run(
            self._net['H'], {self._net['X']: Xval, self._net['T']: Tval, self._net['KP']: 1.0})

        for C in Cs:
            print("---")
            print("---Compute beta...")
            print("~~~balance factor: ", C)
            self._computeB(Htrain, Ttrain, C)

            print("---Prediction...")
            Y = self._predictY(Htrain)
            acc = self._accuracy(Y, Ttrain)

            print("~~~accuracy: %.4f" % (acc), " <--train")

            print("---Start Validing...")
            print("~~~balance factor: ", self._C)

            print("---Prediction...")
            Y = self._predictY(Hval)
            acc = self._accuracy(Y, Tval)
            print("~~~accuracy: %.4f" % (acc), " <--valid")
        print("---Done!")

    def train(self, X, T, C):

        print("===In Training")
        print("---Start Training...")
        N, H, W, C = X.shape
        N, nC = T.shape

        print("---Forward propagation...")
        self._H = self._sess.run(
            self._net['H'], {self._net['X']: X, self._net['T']: T})

        print("---Compute beta...")
        print("~~~balance factor: ", C)
        self._computeB(T, C)

        print("---Prediction...")
        Y = self._predictY()
        acc = self._accuracy(Y, T)

        print("~~~accuracy: %.4f" % (acc), " <--training")

        print("---Done!")

    def test(self, X, T):

        print("===In Testing")
        print("---Start Testing...")
        N, H, W, C = X.shape
        N, nC = T.shape

        print("---Forward propagation...")
        self._H = self._sess.run(
            self._net['H'], {self._net['X']: X, self._net['T']: T})

        print("---Prediction...")
        print("~~~balance factor: ", self._C)
        Y = self._predictY()
        acc = self._accuracy(Y, T)
        print("~~~accuracy: %.4f" % (acc), " <--testing")
        print("---Done!")
