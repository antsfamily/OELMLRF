# 实验1

## 配置

```python
Ns = 10000

Xtrain = Xtrain[0:Ns, :]
Ttrain = Ttrain[0:Ns, :]
Xval = Xval[0:10000, :]
Tval = Tval[0:10000, :]


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

        HO = self._net['P3']
        # HO = self._net['P1']
        # HO = tf.nn.swish(self._net['P1'])
        # HO = tf.nn.lrn(HO)
        # HO = tf.nn.swish(HO)
        # HO = tf.nn.swish(HO)
        # HO = self._net['C1']
        HO = tf.nn.dropout(HO, self._net['KP'])
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
~~~accuracy: 0.9988  <--train
---Start Validing...
~~~balance factor:  100.0
---Prediction...
~~~accuracy: 0.9120  <--valid
---
---Compute beta...
~~~balance factor:  10.0
---Prediction...
~~~accuracy: 0.9840  <--train
---Start Validing...
~~~balance factor:  10.0
---Prediction...
~~~accuracy: 0.8957  <--valid
---
---Compute beta...
~~~balance factor:  5.0
---Prediction...
~~~accuracy: 0.9732  <--train
---Start Validing...
~~~balance factor:  5.0
---Prediction...
~~~accuracy: 0.8805  <--valid
---
---Compute beta...
~~~balance factor:  2.0
---Prediction...
~~~accuracy: 0.9552  <--train
---Start Validing...
~~~balance factor:  2.0
---Prediction...
~~~accuracy: 0.8612  <--valid
---
---Compute beta...
~~~balance factor:  1.5
---Prediction...
~~~accuracy: 0.9491  <--train
---Start Validing...
~~~balance factor:  1.5
---Prediction...
~~~accuracy: 0.8505  <--valid
---
---Compute beta...
~~~balance factor:  0.01
---Prediction...
~~~accuracy: 0.6832  <--train
---Start Validing...
~~~balance factor:  0.01
---Prediction...
~~~accuracy: 0.6355  <--valid
---
---Compute beta...
~~~balance factor:  0.1
---Prediction...
~~~accuracy: 0.8466  <--train
---Start Validing...
~~~balance factor:  0.1
---Prediction...
~~~accuracy: 0.7446  <--valid
---
---Compute beta...
~~~balance factor:  0.2
---Prediction...
~~~accuracy: 0.8805  <--train
---Start Validing...
~~~balance factor:  0.2
---Prediction...
~~~accuracy: 0.7754  <--valid
---
---Compute beta...
~~~balance factor:  0.3
---Prediction...
~~~accuracy: 0.8974  <--train
---Start Validing...
~~~balance factor:  0.3
---Prediction...
~~~accuracy: 0.7935  <--valid
---
---Compute beta...
~~~balance factor:  0.4
---Prediction...
~~~accuracy: 0.9099  <--train
---Start Validing...
~~~balance factor:  0.4
---Prediction...
~~~accuracy: 0.8062  <--valid
---
---Compute beta...
~~~balance factor:  0.5
---Prediction...
~~~accuracy: 0.9169  <--train
---Start Validing...
~~~balance factor:  0.5
---Prediction...
~~~accuracy: 0.8130  <--valid
---
---Compute beta...
~~~balance factor:  0.6
---Prediction...
~~~accuracy: 0.9220  <--train
---Start Validing...
~~~balance factor:  0.6
---Prediction...
~~~accuracy: 0.8190  <--valid
---
---Compute beta...
~~~balance factor:  0.7
---Prediction...
~~~accuracy: 0.9265  <--train
---Start Validing...
~~~balance factor:  0.7
---Prediction...
~~~accuracy: 0.8250  <--valid
---
---Compute beta...
~~~balance factor:  0.8
---Prediction...
~~~accuracy: 0.9310  <--train
---Start Validing...
~~~balance factor:  0.8
---Prediction...
~~~accuracy: 0.8288  <--valid
---
---Compute beta...
~~~balance factor:  0.9
---Prediction...
~~~accuracy: 0.9347  <--train
---Start Validing...
~~~balance factor:  0.9
---Prediction...
~~~accuracy: 0.8334  <--valid
---Done!
[Finished in 519.9s]

```

# 实验2

## 配置

```python
Ns = 10000

Xtrain = Xtrain[0:Ns, :]
Ttrain = Ttrain[0:Ns, :]
Xval = Xval[0:10000, :]
Tval = Tval[0:10000, :]
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
~~~accuracy: 0.8401  <--valid
---
---Compute beta...
~~~balance factor:  10.0
---Prediction...
~~~accuracy: 1.0000  <--train
---Start Validing...
~~~balance factor:  10.0
---Prediction...
~~~accuracy: 0.8987  <--valid
---
---Compute beta...
~~~balance factor:  5.0
---Prediction...
~~~accuracy: 1.0000  <--train
---Start Validing...
~~~balance factor:  5.0
---Prediction...
~~~accuracy: 0.9104  <--valid
---
---Compute beta...
~~~balance factor:  2.0
---Prediction...
~~~accuracy: 1.0000  <--train
---Start Validing...
~~~balance factor:  2.0
---Prediction...
~~~accuracy: 0.9168  <--valid
---
---Compute beta...
~~~balance factor:  1.5
---Prediction...
~~~accuracy: 1.0000  <--train
---Start Validing...
~~~balance factor:  1.5
---Prediction...
~~~accuracy: 0.9189  <--valid
---
---Compute beta...
~~~balance factor:  0.01
---Prediction...
~~~accuracy: 0.9659  <--train
---Start Validing...
~~~balance factor:  0.01
---Prediction...
~~~accuracy: 0.8840  <--valid
---
---Compute beta...
~~~balance factor:  0.1
---Prediction...
~~~accuracy: 0.9947  <--train
---Start Validing...
~~~balance factor:  0.1
---Prediction...
~~~accuracy: 0.9180  <--valid
---
---Compute beta...
~~~balance factor:  0.2
---Prediction...
~~~accuracy: 0.9973  <--train
---Start Validing...
~~~balance factor:  0.2
---Prediction...
~~~accuracy: 0.9222  <--valid
---
---Compute beta...
~~~balance factor:  0.3
---Prediction...
~~~accuracy: 0.9988  <--train
---Start Validing...
~~~balance factor:  0.3
---Prediction...
~~~accuracy: 0.9230  <--valid
---
---Compute beta...
~~~balance factor:  0.4
---Prediction...
~~~accuracy: 0.9992  <--train
---Start Validing...
~~~balance factor:  0.4
---Prediction...
~~~accuracy: 0.9227  <--valid
---
---Compute beta...
~~~balance factor:  0.5
---Prediction...
~~~accuracy: 0.9993  <--train
---Start Validing...
~~~balance factor:  0.5
---Prediction...
~~~accuracy: 0.9224  <--valid
---
---Compute beta...
~~~balance factor:  0.6
---Prediction...
~~~accuracy: 0.9995  <--train
---Start Validing...
~~~balance factor:  0.6
---Prediction...
~~~accuracy: 0.9221  <--valid
---
---Compute beta...
~~~balance factor:  0.7
---Prediction...
~~~accuracy: 0.9995  <--train
---Start Validing...
~~~balance factor:  0.7
---Prediction...
~~~accuracy: 0.9230  <--valid
---
---Compute beta...
~~~balance factor:  0.8
---Prediction...
~~~accuracy: 0.9999  <--train
---Start Validing...
~~~balance factor:  0.8
---Prediction...
~~~accuracy: 0.9217  <--valid
---
---Compute beta...
~~~balance factor:  0.9
---Prediction...
~~~accuracy: 1.0000  <--train
---Start Validing...
~~~balance factor:  0.9
---Prediction...
~~~accuracy: 0.9209  <--valid
---Done!
[Finished in 649.4s]
```


# 实验3

## 配置

```python
Ns = 10000

Xtrain = Xtrain[0:Ns, :]
Ttrain = Ttrain[0:Ns, :]
Xval = Xval[0:10000, :]
Tval = Tval[0:10000, :]
```


## 结果


```


```
