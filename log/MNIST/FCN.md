






# 实验1

## 配置

```python
    Ns = 20000

    Xtrain = Xtrain[0:Ns, :] / 255.0
    Ttrain = Ttrain[0:Ns, :]
    Xval = Xval[0:10000, :] / 255.0
    Tval = Tval[0:10000, :]
```

C4 30000个滤波器

```python
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
            name='C1', shape=[5, 5, self._insize[2], 32],
            dtype=dtype, initializer=InitA, trainable=False)

        self._W['C2'] = tf.get_variable(
            name='C2', shape=[3, 3, 32, 32],
            dtype=dtype, initializer=InitA, trainable=False)

        self._W['C3'] = tf.get_variable(
            name='C3', shape=[3, 3, 32, 32],
            dtype=dtype, initializer=InitA, trainable=False)

        self._W['C4'] = tf.get_variable(
            name='C4', shape=[14, 14, 32, 30000],
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

        self._net['C4'] = tf.nn.conv2d(
            self._net['P3'], self._W['C4'],
            strides=[1, 1, 1, 1],  # [1, sH, sW, 1]
            padding="VALID", data_format='NHWC',
            dilations=[1, 1, 1, 1], name='C4')

        self._net['C4'] = tf.nn.swish(self._net['C4'])

        HO = self._net['C4']
        # HO = self._net['P3']
        # HO = self._net['P1']
        # HO = tf.nn.swish(self._net['P1'])
        # HO = tf.nn.lrn(HO)
        # HO = tf.nn.swish(HO)
        # HO = tf.nn.swish(HO)
        # HO = self._net['C1']
        # HO = tf.nn.dropout(HO, self._net['KP'])
        # N, H, W, C = HO.get_shape()

        # H --> N - H * W * C
        # self._net['H'] = tf.reshape(HO, shape=[-1, H * W * C], name='H')
        # self._net['H'] = tf.reshape(HO, shape=[-1, self._L], name='H')
        self._net['H'] = tf.reshape(HO, shape=[-1, 30000], name='H')
```

## 结果

```
===In Training Validing
---Start Training...
---Forward propagation...
---
---Compute beta...
~~~balance factor:  100.0
---Prediction...
~~~accuracy: 0.0969  <--train
---Start Validing...
~~~balance factor:  100.0
---Prediction...
~~~accuracy: 0.0980  <--valid
---
---Compute beta...
~~~balance factor:  10.0
---Prediction...
~~~accuracy: 0.0969  <--train
---Start Validing...
~~~balance factor:  10.0
---Prediction...
~~~accuracy: 0.0980  <--valid
---
---Compute beta...
~~~balance factor:  5.0
---Prediction...
~~~accuracy: 0.0969  <--train
---Start Validing...
~~~balance factor:  5.0
---Prediction...
~~~accuracy: 0.0980  <--valid
---
---Compute beta...
~~~balance factor:  2.0
---Prediction...
~~~accuracy: 0.0969  <--train
---Start Validing...
~~~balance factor:  2.0
---Prediction...
~~~accuracy: 0.0980  <--valid
---
---Compute beta...
~~~balance factor:  1.5
---Prediction...
~~~accuracy: 0.0969  <--train
---Start Validing...
~~~balance factor:  1.5
---Prediction...
~~~accuracy: 0.0980  <--valid
---
---Compute beta...
~~~balance factor:  0.01
---Prediction...
~~~accuracy: 0.0969  <--train
---Start Validing...
~~~balance factor:  0.01
---Prediction...
~~~accuracy: 0.0980  <--valid
---
---Compute beta...
~~~balance factor:  0.1
---Prediction...
~~~accuracy: 0.0969  <--train
---Start Validing...
~~~balance factor:  0.1
---Prediction...
~~~accuracy: 0.0980  <--valid
---
---Compute beta...
~~~balance factor:  0.2
---Prediction...
~~~accuracy: 0.0969  <--train
---Start Validing...
~~~balance factor:  0.2
---Prediction...
~~~accuracy: 0.0980  <--valid
---
---Compute beta...
~~~balance factor:  0.3
---Prediction...
~~~accuracy: 0.0969  <--train
---Start Validing...
~~~balance factor:  0.3
---Prediction...
~~~accuracy: 0.0980  <--valid
---
---Compute beta...
~~~balance factor:  0.4
---Prediction...
~~~accuracy: 0.0969  <--train
---Start Validing...
~~~balance factor:  0.4
---Prediction...
~~~accuracy: 0.0980  <--valid
---
---Compute beta...
~~~balance factor:  0.5
---Prediction...
~~~accuracy: 0.0969  <--train
---Start Validing...
~~~balance factor:  0.5
---Prediction...
~~~accuracy: 0.0980  <--valid
---
---Compute beta...
~~~balance factor:  0.6
---Prediction...
~~~accuracy: 0.0969  <--train
---Start Validing...
~~~balance factor:  0.6
---Prediction...
~~~accuracy: 0.0980  <--valid
---
---Compute beta...
~~~balance factor:  0.7
---Prediction...
~~~accuracy: 0.0969  <--train
---Start Validing...
~~~balance factor:  0.7
---Prediction...
~~~accuracy: 0.0980  <--valid
---
---Compute beta...
~~~balance factor:  0.8
---Prediction...
~~~accuracy: 0.0969  <--train
---Start Validing...
~~~balance factor:  0.8
---Prediction...
~~~accuracy: 0.0980  <--valid
---
---Compute beta...
~~~balance factor:  0.9
---Prediction...
~~~accuracy: 0.0969  <--train
---Start Validing...
~~~balance factor:  0.9
---Prediction...
~~~accuracy: 0.0980  <--valid
---Done!
```


# 实验2

## 配置

```
C4 10000个滤波器
```

## 结果

```
===In Training Validing
---Start Training...
---Forward propagation...
---
---Compute beta...
~~~balance factor:  100.0
---Prediction...
~~~accuracy: 0.0969  <--train
---Start Validing...
~~~balance factor:  100.0
---Prediction...
~~~accuracy: 0.0980  <--valid
---
---Compute beta...
~~~balance factor:  10.0
---Prediction...
~~~accuracy: 0.0969  <--train
---Start Validing...
~~~balance factor:  10.0
---Prediction...
~~~accuracy: 0.0980  <--valid
---
---Compute beta...
~~~balance factor:  5.0
---Prediction...
~~~accuracy: 0.0969  <--train
---Start Validing...
~~~balance factor:  5.0
---Prediction...
~~~accuracy: 0.0980  <--valid
---
---Compute beta...
~~~balance factor:  2.0
---Prediction...
~~~accuracy: 0.0969  <--train
---Start Validing...
~~~balance factor:  2.0
---Prediction...
~~~accuracy: 0.0980  <--valid
---
---Compute beta...
~~~balance factor:  1.5
---Prediction...
~~~accuracy: 0.0969  <--train
---Start Validing...
~~~balance factor:  1.5
---Prediction...
~~~accuracy: 0.0980  <--valid
---
---Compute beta...
~~~balance factor:  0.01
---Prediction...
~~~accuracy: 0.0969  <--train
---Start Validing...
~~~balance factor:  0.01
---Prediction...
~~~accuracy: 0.0980  <--valid
---
---Compute beta...
~~~balance factor:  0.1
---Prediction...
~~~accuracy: 0.0969  <--train
---Start Validing...
~~~balance factor:  0.1
---Prediction...
~~~accuracy: 0.0980  <--valid
---
---Compute beta...
~~~balance factor:  0.2
---Prediction...
~~~accuracy: 0.0969  <--train
---Start Validing...
~~~balance factor:  0.2
---Prediction...
~~~accuracy: 0.0980  <--valid
---
---Compute beta...
~~~balance factor:  0.3
---Prediction...
~~~accuracy: 0.0969  <--train
---Start Validing...
~~~balance factor:  0.3
---Prediction...
~~~accuracy: 0.0980  <--valid
---
---Compute beta...
~~~balance factor:  0.4
---Prediction...
~~~accuracy: 0.0969  <--train
---Start Validing...
~~~balance factor:  0.4
---Prediction...
~~~accuracy: 0.0980  <--valid
---
---Compute beta...
~~~balance factor:  0.5
---Prediction...
~~~accuracy: 0.0969  <--train
---Start Validing...
~~~balance factor:  0.5
---Prediction...
~~~accuracy: 0.0980  <--valid
---
---Compute beta...
~~~balance factor:  0.6
---Prediction...
~~~accuracy: 0.0969  <--train
---Start Validing...
~~~balance factor:  0.6
---Prediction...
~~~accuracy: 0.0980  <--valid
---
---Compute beta...
~~~balance factor:  0.7
---Prediction...
~~~accuracy: 0.0969  <--train
---Start Validing...
~~~balance factor:  0.7
---Prediction...
~~~accuracy: 0.0980  <--valid
---
---Compute beta...
~~~balance factor:  0.8
---Prediction...
~~~accuracy: 0.0969  <--train
---Start Validing...
~~~balance factor:  0.8
---Prediction...
~~~accuracy: 0.0980  <--valid
---
---Compute beta...
~~~balance factor:  0.9
---Prediction...
~~~accuracy: 0.0969  <--train
---Start Validing...
~~~balance factor:  0.9
---Prediction...
~~~accuracy: 0.0980  <--valid
---Done!
[Finished in 663.4s]
```



# 实验3

## 配置

```

```

## 结果

```

```


# 实验4

## 配置

```

```

## 结果

```

```
