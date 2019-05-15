# 实验1

## 配置

```
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

        HO = self._net['P1']
        # HO = self._net['P1']
        # HO = tf.nn.swish(self._net['P1'])
        # HO = tf.nn.lrn(HO)
        HO = tf.nn.swish(HO)
        # HO = tf.nn.swish(HO)
        # HO = self._net['C1']
        N, H, W, C = HO.get_shape()
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
~~~accuracy: 0.9998  <--train
---Start Validing...
~~~balance factor:  100.0
---Prediction...
~~~accuracy: 0.9735  <--valid
---
---Compute beta...
~~~balance factor:  10.0
---Prediction...
~~~accuracy: 0.9988  <--train
---Start Validing...
~~~balance factor:  10.0
---Prediction...
~~~accuracy: 0.9791  <--valid
---
---Compute beta...
~~~balance factor:  5.0
---Prediction...
~~~accuracy: 0.9982  <--train
---Start Validing...
~~~balance factor:  5.0
---Prediction...
~~~accuracy: 0.9794  <--valid
---
---Compute beta...
~~~balance factor:  2.0
---Prediction...
~~~accuracy: 0.9964  <--train
---Start Validing...
~~~balance factor:  2.0
---Prediction...
~~~accuracy: 0.9802  <--valid
---
---Compute beta...
~~~balance factor:  1.5
---Prediction...
~~~accuracy: 0.9956  <--train
---Start Validing...
~~~balance factor:  1.5
---Prediction...
~~~accuracy: 0.9805  <--valid
---
---Compute beta...
~~~balance factor:  0.01
---Prediction...
~~~accuracy: 0.9694  <--train
---Start Validing...
~~~balance factor:  0.01
---Prediction...
~~~accuracy: 0.9681  <--valid
---
---Compute beta...
~~~balance factor:  0.1
---Prediction...
~~~accuracy: 0.9839  <--train
---Start Validing...
~~~balance factor:  0.1
---Prediction...
~~~accuracy: 0.9779  <--valid
---
---Compute beta...
~~~balance factor:  0.2
---Prediction...
~~~accuracy: 0.9870  <--train
---Start Validing...
~~~balance factor:  0.2
---Prediction...
~~~accuracy: 0.9800  <--valid
---
---Compute beta...
~~~balance factor:  0.3
---Prediction...
~~~accuracy: 0.9889  <--train
---Start Validing...
~~~balance factor:  0.3
---Prediction...
~~~accuracy: 0.9812  <--valid
---
---Compute beta...
~~~balance factor:  0.4
---Prediction...
~~~accuracy: 0.9903  <--train
---Start Validing...
~~~balance factor:  0.4
---Prediction...
~~~accuracy: 0.9816  <--valid
---
---Compute beta...
~~~balance factor:  0.5
---Prediction...
~~~accuracy: 0.9911  <--train
---Start Validing...
~~~balance factor:  0.5
---Prediction...
~~~accuracy: 0.9817  <--valid
---
---Compute beta...
~~~balance factor:  0.6
---Prediction...
~~~accuracy: 0.9921  <--train
---Start Validing...
~~~balance factor:  0.6
---Prediction...
~~~accuracy: 0.9819  <--valid
---
---Compute beta...
~~~balance factor:  0.7
---Prediction...
~~~accuracy: 0.9930  <--train
---Start Validing...
~~~balance factor:  0.7
---Prediction...
~~~accuracy: 0.9818  <--valid
---
---Compute beta...
~~~balance factor:  0.8
---Prediction...
~~~accuracy: 0.9937  <--train
---Start Validing...
~~~balance factor:  0.8
---Prediction...
~~~accuracy: 0.9814  <--valid
---
---Compute beta...
~~~balance factor:  0.9
---Prediction...
~~~accuracy: 0.9940  <--train
---Start Validing...
~~~balance factor:  0.9
---Prediction...
~~~accuracy: 0.9811  <--valid
---Done!
[Finished in 1804.8s]
```

# 实验2

## 配置

```
        print("===Build ELMLRF...")
        print("---Set random seed: " + str(randseed))
        tf.set_random_seed(randseed)

        print("---Graph node...")
        self._net['X'] = tf.placeholder(
            dtype=dtype, shape=[None] + self._insize, name='inputs')
        self._net['T'] = tf.placeholder(
            dtype=dtype, shape=[None] + self._outsize, name='targets')

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
        N, H, W, C = HO.get_shape()
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
~~~accuracy: 1.0000  <--train
---Start Validing...
~~~balance factor:  100.0
---Prediction...
~~~accuracy: 0.9679  <--valid
---
---Compute beta...
~~~balance factor:  10.0
---Prediction...
~~~accuracy: 0.9996  <--train
---Start Validing...
~~~balance factor:  10.0
---Prediction...
~~~accuracy: 0.9769  <--valid
---
---Compute beta...
~~~balance factor:  5.0
---Prediction...
~~~accuracy: 0.9991  <--train
---Start Validing...
~~~balance factor:  5.0
---Prediction...
~~~accuracy: 0.9777  <--valid
---
---Compute beta...
~~~balance factor:  2.0
---Prediction...
~~~accuracy: 0.9987  <--train
---Start Validing...
~~~balance factor:  2.0
---Prediction...
~~~accuracy: 0.9797  <--valid
---
---Compute beta...
~~~balance factor:  1.5
---Prediction...
~~~accuracy: 0.9984  <--train
---Start Validing...
~~~balance factor:  1.5
---Prediction...
~~~accuracy: 0.9803  <--valid
---
---Compute beta...
~~~balance factor:  0.01
---Prediction...
~~~accuracy: 0.9779  <--train
---Start Validing...
~~~balance factor:  0.01
---Prediction...
~~~accuracy: 0.9737  <--valid
---
---Compute beta...
~~~balance factor:  0.1
---Prediction...
~~~accuracy: 0.9898  <--train
---Start Validing...
~~~balance factor:  0.1
---Prediction...
~~~accuracy: 0.9804  <--valid
---
---Compute beta...
~~~balance factor:  0.2
---Prediction...
~~~accuracy: 0.9930  <--train
---Start Validing...
~~~balance factor:  0.2
---Prediction...
~~~accuracy: 0.9809  <--valid
---
---Compute beta...
~~~balance factor:  0.3
---Prediction...
~~~accuracy: 0.9946  <--train
---Start Validing...
~~~balance factor:  0.3
---Prediction...
~~~accuracy: 0.9807  <--valid
---
---Compute beta...
~~~balance factor:  0.4
---Prediction...
~~~accuracy: 0.9956  <--train
---Start Validing...
~~~balance factor:  0.4
---Prediction...
~~~accuracy: 0.9807  <--valid
---
---Compute beta...
~~~balance factor:  0.5
---Prediction...
~~~accuracy: 0.9962  <--train
---Start Validing...
~~~balance factor:  0.5
---Prediction...
~~~accuracy: 0.9808  <--valid
---
---Compute beta...
~~~balance factor:  0.6
---Prediction...
~~~accuracy: 0.9968  <--train
---Start Validing...
~~~balance factor:  0.6
---Prediction...
~~~accuracy: 0.9804  <--valid
---
---Compute beta...
~~~balance factor:  0.7
---Prediction...
~~~accuracy: 0.9971  <--train
---Start Validing...
~~~balance factor:  0.7
---Prediction...
~~~accuracy: 0.9805  <--valid
---
---Compute beta...
~~~balance factor:  0.8
---Prediction...
~~~accuracy: 0.9972  <--train
---Start Validing...
~~~balance factor:  0.8
---Prediction...
~~~accuracy: 0.9807  <--valid
---
---Compute beta...
~~~balance factor:  0.9
---Prediction...
~~~accuracy: 0.9975  <--train
---Start Validing...
~~~balance factor:  0.9
---Prediction...
~~~accuracy: 0.9807  <--valid
---Done!
[Finished in 1673.5s]
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
