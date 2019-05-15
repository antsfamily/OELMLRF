# 实验1

## 配置

```
Ns = 20000

Xtrain = Xtrain[0:Ns, :] / 255.0
Ttrain = Ttrain[0:Ns, :]
Xval = Xval[0:10000, :] / 255.0
Tval = Tval[0:10000, :]
randomseed = 0

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
    name='C1', shape=[5, 5, self._insize[2], 16],
    dtype=dtype, initializer=InitA, trainable=False)

self._W['C2'] = tf.get_variable(
    name='C2', shape=[3, 3, 16, 16],
    dtype=dtype, initializer=InitA, trainable=False)

self._W['C3'] = tf.get_variable(
    name='C3', shape=[3, 3, 16, 32],
    dtype=dtype, initializer=InitA, trainable=False)

# ----------------------------gen net
print("---Construct ELMLRF Net...")
self._net['C1'] = tf.nn.conv2d(
    self._net['X'], self._W['C1'],
    strides=[1, 1, 1, 1],  # [1, sH, sW, 1]
    padding="VALID", data_format='NHWC',
    dilations=[1, 1, 1, 1], name='C1')

self._net['C1'] = tf.nn.relu(self._net['C1'])
self._net['C1'] = tf.nn.lrn(self._net['C1'])

# self._net['P1'] = tf.nn.max_pool(
self._net['P1'] = tt.nn.square_root_pool(
    self._net['C1'],
    ksize=[1, 3, 3, 1],  # [1, kH, kW, 1]
    strides=[1, 1, 1, 1],  # [1, sH, sW, 1]
    padding="VALID", data_format='NHWC', name='P1')

self._net['C2'] = tf.nn.conv2d(
    self._net['P1'], self._W['C2'],
    strides=[1, 1, 1, 1],  # [1, sH, sW, 1]
    padding="VALID", data_format='NHWC',
    dilations=[1, 1, 1, 1], name='C2')

self._net['C2'] = tf.nn.relu(self._net['C2'])
self._net['C2'] = tf.nn.lrn(self._net['C2'])

# self._net['P2'] = tf.nn.max_pool(
self._net['P2'] = tt.nn.square_root_pool(
    self._net['C2'],
    ksize=[1, 3, 3, 1],  # [1, kH, kW, 1]
    strides=[1, 1, 1, 1],  # [1, sH, sW, 1]
    padding="VALID", data_format='NHWC', name='P2')

self._net['C3'] = tf.nn.conv2d(
    self._net['P2'], self._W['C3'],
    strides=[1, 1, 1, 1],  # [1, sH, sW, 1]
    padding="VALID", data_format='NHWC',
    dilations=[1, 1, 1, 1], name='C3')

self._net['C3'] = tf.nn.relu(self._net['C3'])
self._net['C3'] = tf.nn.lrn(self._net['C3'])

# self._net['P3'] = tf.nn.max_pool(
self._net['P3'] = tt.nn.square_root_pool(
    self._net['C3'],
    ksize=[1, 3, 3, 1],  # [1, kH, kW, 1]
    strides=[1, 1, 1, 1],  # [1, sH, sW, 1]
    padding="VALID", data_format='NHWC', name='P3')

HO = self._net['P3']
# HO = self._net['P1']
# HO = tf.nn.relu(self._net['P1'])
# HO = tf.nn.lrn(HO)
# HO = self._net['C1']
N, H, W, C = HO.get_shape()

```

## 结果


```
===Build ELMLRF...
---Set random seed: 0
---Graph node...
---Generates ELMLRF random weights(orthogonal)...
---Construct ELMLRF Net...
~~~X -->  (?, 28, 28, 1)
~~~T -->  (?, 10)
~~~H -->  (?, 6272)
~~~L -->  6272
~~~HO -->  (?, 14, 14, 32)
---Initialize variables...
C1 <tf.Variable 'C1:0' shape=(5, 5, 1, 16) dtype=float32_ref>
C3 <tf.Variable 'C3:0' shape=(3, 3, 16, 32) dtype=float32_ref>
C2 <tf.Variable 'C2:0' shape=(3, 3, 16, 16) dtype=float32_ref>
WARNING:tensorflow:From /usr/local/lib/python3.5/dist-packages/tensorflow/python/util/tf_should_use.py:118: initialize_variables (from tensorflow.python.ops.variables) is deprecated and will be removed after 2017-03-02.
Instructions for updating:
Use `tf.variables_initializer` instead.
2019-05-14 16:31:27.783804: I tensorflow/core/kernels/cuda_solvers.cc:159] Creating CudaSolver handles for stream 0x5053e40
===Done!
===In Training Validing
---Start Training...
---Forward propagation...
---
---Compute beta...
~~~balance factor:  0.001
---Prediction...
~~~accuracy: 0.8960  <--train
---Start Validing...
~~~balance factor:  0.001
---Prediction...
~~~accuracy: 0.9038  <--valid
---
---Compute beta...
~~~balance factor:  0.01
---Prediction...
~~~accuracy: 0.9490  <--train
---Start Validing...
~~~balance factor:  0.01
---Prediction...
~~~accuracy: 0.9516  <--valid
---
---Compute beta...
~~~balance factor:  0.1
---Prediction...
~~~accuracy: 0.9712  <--train
---Start Validing...
~~~balance factor:  0.1
---Prediction...
~~~accuracy: 0.9707  <--valid
---
---Compute beta...
~~~balance factor:  0.2
---Prediction...
~~~accuracy: 0.9750  <--train
---Start Validing...
~~~balance factor:  0.2
---Prediction...
~~~accuracy: 0.9743  <--valid
---
---Compute beta...
~~~balance factor:  0.3
---Prediction...
~~~accuracy: 0.9764  <--train
---Start Validing...
~~~balance factor:  0.3
---Prediction...
~~~accuracy: 0.9757  <--valid
---
---Compute beta...
~~~balance factor:  0.4
---Prediction...
~~~accuracy: 0.9771  <--train
---Start Validing...
~~~balance factor:  0.4
---Prediction...
~~~accuracy: 0.9764  <--valid
---
---Compute beta...
~~~balance factor:  0.5
---Prediction...
~~~accuracy: 0.9781  <--train
---Start Validing...
~~~balance factor:  0.5
---Prediction...
~~~accuracy: 0.9770  <--valid
---
---Compute beta...
~~~balance factor:  0.6
---Prediction...
~~~accuracy: 0.9787  <--train
---Start Validing...
~~~balance factor:  0.6
---Prediction...
~~~accuracy: 0.9775  <--valid
---
---Compute beta...
~~~balance factor:  0.7
---Prediction...
~~~accuracy: 0.9791  <--train
---Start Validing...
~~~balance factor:  0.7
---Prediction...
~~~accuracy: 0.9775  <--valid
---
---Compute beta...
~~~balance factor:  0.8
---Prediction...
~~~accuracy: 0.9798  <--train
---Start Validing...
~~~balance factor:  0.8
---Prediction...
~~~accuracy: 0.9779  <--valid
---
---Compute beta...
~~~balance factor:  0.9
---Prediction...
~~~accuracy: 0.9799  <--train
---Start Validing...
~~~balance factor:  0.9
---Prediction...
~~~accuracy: 0.9781  <--valid
---Done!
[Finished in 179.0s]

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

        self._net['C1'] = tf.nn.relu(self._net['C1'])
        # self._net['C1'] = tf.nn.lrn(self._net['C1'])

        # self._net['P1'] = tf.nn.max_pool(
        self._net['P1'] = tt.nn.square_root_pool(
            self._net['C1'],
            ksize=[1, 3, 3, 1],  # [1, kH, kW, 1]
            strides=[1, 1, 1, 1],  # [1, sH, sW, 1]
            padding="VALID", data_format='NHWC', name='P1')

        self._net['C2'] = tf.nn.conv2d(
            self._net['P1'], self._W['C2'],
            strides=[1, 1, 1, 1],  # [1, sH, sW, 1]
            padding="VALID", data_format='NHWC',
            dilations=[1, 1, 1, 1], name='C2')

        self._net['C2'] = tf.nn.relu(self._net['C2'])
        # self._net['C2'] = tf.nn.lrn(self._net['C2'])

        # self._net['P2'] = tf.nn.max_pool(
        self._net['P2'] = tt.nn.square_root_pool(
            self._net['C2'],
            ksize=[1, 3, 3, 1],  # [1, kH, kW, 1]
            strides=[1, 1, 1, 1],  # [1, sH, sW, 1]
            padding="VALID", data_format='NHWC', name='P2')

        self._net['C3'] = tf.nn.conv2d(
            self._net['P2'], self._W['C3'],
            strides=[1, 1, 1, 1],  # [1, sH, sW, 1]
            padding="VALID", data_format='NHWC',
            dilations=[1, 1, 1, 1], name='C3')

        self._net['C3'] = tf.nn.relu(self._net['C3'])
        # self._net['C3'] = tf.nn.lrn(self._net['C3'])

        # self._net['P3'] = tf.nn.max_pool(
        self._net['P3'] = tt.nn.square_root_pool(
            self._net['C3'],
            ksize=[1, 3, 3, 1],  # [1, kH, kW, 1]
            strides=[1, 1, 1, 1],  # [1, sH, sW, 1]
            padding="VALID", data_format='NHWC', name='P3')

        HO = self._net['P3']
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
    ~~~accuracy: 0.9944  <--train
    ---Start Validing...
    ~~~balance factor:  100.0
    ---Prediction...
    ~~~accuracy: 0.9779  <--valid
    ---
    ---Compute beta...
    ~~~balance factor:  10.0
    ---Prediction...
    ~~~accuracy: 0.9891  <--train
    ---Start Validing...
    ~~~balance factor:  10.0
    ---Prediction...
    ~~~accuracy: 0.9799  <--valid
    ---
    ---Compute beta...
    ~~~balance factor:  5.0
    ---Prediction...
    ~~~accuracy: 0.9875  <--train
    ---Start Validing...
    ~~~balance factor:  5.0
    ---Prediction...
    ~~~accuracy: 0.9802  <--valid
    ---
    ---Compute beta...
    ~~~balance factor:  2.0
    ---Prediction...
    ~~~accuracy: 0.9852  <--train
    ---Start Validing...
    ~~~balance factor:  2.0
    ---Prediction...
    ~~~accuracy: 0.9805  <--valid
    ---
    ---Compute beta...
    ~~~balance factor:  1.5
    ---Prediction...
    ~~~accuracy: 0.9840  <--train
    ---Start Validing...
    ~~~balance factor:  1.5
    ---Prediction...
    ~~~accuracy: 0.9804  <--valid
    ---
    ---Compute beta...
    ~~~balance factor:  0.01
    ---Prediction...
    ~~~accuracy: 0.9589  <--train
    ---Start Validing...
    ~~~balance factor:  0.01
    ---Prediction...
    ~~~accuracy: 0.9621  <--valid
    ---
    ---Compute beta...
    ~~~balance factor:  0.1
    ---Prediction...
    ~~~accuracy: 0.9754  <--train
    ---Start Validing...
    ~~~balance factor:  0.1
    ---Prediction...
    ~~~accuracy: 0.9765  <--valid
    ---
    ---Compute beta...
    ~~~balance factor:  0.2
    ---Prediction...
    ~~~accuracy: 0.9778  <--train
    ---Start Validing...
    ~~~balance factor:  0.2
    ---Prediction...
    ~~~accuracy: 0.9788  <--valid
    ---
    ---Compute beta...
    ~~~balance factor:  0.3
    ---Prediction...
    ~~~accuracy: 0.9790  <--train
    ---Start Validing...
    ~~~balance factor:  0.3
    ---Prediction...
    ~~~accuracy: 0.9792  <--valid
    ---
    ---Compute beta...
    ~~~balance factor:  0.4
    ---Prediction...
    ~~~accuracy: 0.9798  <--train
    ---Start Validing...
    ~~~balance factor:  0.4
    ---Prediction...
    ~~~accuracy: 0.9796  <--valid
    ---
    ---Compute beta...
    ~~~balance factor:  0.5
    ---Prediction...
    ~~~accuracy: 0.9804  <--train
    ---Start Validing...
    ~~~balance factor:  0.5
    ---Prediction...
    ~~~accuracy: 0.9797  <--valid
    ---
    ---Compute beta...
    ~~~balance factor:  0.6
    ---Prediction...
    ~~~accuracy: 0.9808  <--train
    ---Start Validing...
    ~~~balance factor:  0.6
    ---Prediction...
    ~~~accuracy: 0.9802  <--valid
    ---
    ---Compute beta...
    ~~~balance factor:  0.7
    ---Prediction...
    ~~~accuracy: 0.9816  <--train
    ---Start Validing...
    ~~~balance factor:  0.7
    ---Prediction...
    ~~~accuracy: 0.9803  <--valid
    ---
    ---Compute beta...
    ~~~balance factor:  0.8
    ---Prediction...
    ~~~accuracy: 0.9820  <--train
    ---Start Validing...
    ~~~balance factor:  0.8
    ---Prediction...
    ~~~accuracy: 0.9803  <--valid
    ---
    ---Compute beta...
    ~~~balance factor:  0.9
    ---Prediction...
    ~~~accuracy: 0.9822  <--train
    ---Start Validing...
    ~~~balance factor:  0.9
    ---Prediction...
    ~~~accuracy: 0.9803  <--valid
    ---Done!
    [Finished in 271.6s]
```

# 实验3

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

        self._net['C1'] = tf.nn.relu(self._net['C1'])

        # self._net['P1'] = tf.nn.max_pool(
        self._net['P1'] = tt.nn.square_root_pool(
            self._net['C1'],
            ksize=[1, 3, 3, 1],  # [1, kH, kW, 1]
            strides=[1, 1, 1, 1],  # [1, sH, sW, 1]
            padding="VALID", data_format='NHWC', name='P1')

        self._net['P1'] = tf.nn.lrn(self._net['P1'])
        self._net['C2'] = tf.nn.conv2d(
            self._net['P1'], self._W['C2'],
            strides=[1, 1, 1, 1],  # [1, sH, sW, 1]
            padding="VALID", data_format='NHWC',
            dilations=[1, 1, 1, 1], name='C2')

        self._net['C2'] = tf.nn.relu(self._net['C2'])

        # self._net['P2'] = tf.nn.max_pool(
        self._net['P2'] = tt.nn.square_root_pool(
            self._net['C2'],
            ksize=[1, 3, 3, 1],  # [1, kH, kW, 1]
            strides=[1, 1, 1, 1],  # [1, sH, sW, 1]
            padding="VALID", data_format='NHWC', name='P2')

        self._net['P2'] = tf.nn.lrn(self._net['P2'])

        self._net['C3'] = tf.nn.conv2d(
            self._net['P2'], self._W['C3'],
            strides=[1, 1, 1, 1],  # [1, sH, sW, 1]
            padding="VALID", data_format='NHWC',
            dilations=[1, 1, 1, 1], name='C3')

        self._net['C3'] = tf.nn.relu(self._net['C3'])

        # self._net['P3'] = tf.nn.max_pool(
        self._net['P3'] = tt.nn.square_root_pool(
            self._net['C3'],
            ksize=[1, 3, 3, 1],  # [1, kH, kW, 1]
            strides=[1, 1, 1, 1],  # [1, sH, sW, 1]
            padding="VALID", data_format='NHWC', name='P3')

        # self._net['P3'] = tf.nn.lrn(self._net['P3'])

        HO = self._net['P3']
        # HO = self._net['P1']
        # HO = tf.nn.relu(self._net['P1'])
        # HO = tf.nn.lrn(HO)
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
~~~accuracy: 0.9928  <--train
---Start Validing...
~~~balance factor:  100.0
---Prediction...
~~~accuracy: 0.9775  <--valid
---
---Compute beta...
~~~balance factor:  10.0
---Prediction...
~~~accuracy: 0.9872  <--train
---Start Validing...
~~~balance factor:  10.0
---Prediction...
~~~accuracy: 0.9788  <--valid
---
---Compute beta...
~~~balance factor:  5.0
---Prediction...
~~~accuracy: 0.9856  <--train
---Start Validing...
~~~balance factor:  5.0
---Prediction...
~~~accuracy: 0.9793  <--valid
---
---Compute beta...
~~~balance factor:  2.0
---Prediction...
~~~accuracy: 0.9827  <--train
---Start Validing...
~~~balance factor:  2.0
---Prediction...
~~~accuracy: 0.9793  <--valid
---
---Compute beta...
~~~balance factor:  1.5
---Prediction...
~~~accuracy: 0.9816  <--train
---Start Validing...
~~~balance factor:  1.5
---Prediction...
~~~accuracy: 0.9791  <--valid
---
---Compute beta...
~~~balance factor:  0.01
---Prediction...
~~~accuracy: 0.9488  <--train
---Start Validing...
~~~balance factor:  0.01
---Prediction...
~~~accuracy: 0.9512  <--valid
---
---Compute beta...
~~~balance factor:  0.1
---Prediction...
~~~accuracy: 0.9708  <--train
---Start Validing...
~~~balance factor:  0.1
---Prediction...
~~~accuracy: 0.9713  <--valid
---
---Compute beta...
~~~balance factor:  0.2
---Prediction...
~~~accuracy: 0.9740  <--train
---Start Validing...
~~~balance factor:  0.2
---Prediction...
~~~accuracy: 0.9748  <--valid
---
---Compute beta...
~~~balance factor:  0.3
---Prediction...
~~~accuracy: 0.9761  <--train
---Start Validing...
~~~balance factor:  0.3
---Prediction...
~~~accuracy: 0.9761  <--valid
---
---Compute beta...
~~~balance factor:  0.4
---Prediction...
~~~accuracy: 0.9775  <--train
---Start Validing...
~~~balance factor:  0.4
---Prediction...
~~~accuracy: 0.9773  <--valid
---
---Compute beta...
~~~balance factor:  0.5
---Prediction...
~~~accuracy: 0.9782  <--train
---Start Validing...
~~~balance factor:  0.5
---Prediction...
~~~accuracy: 0.9777  <--valid
---
---Compute beta...
~~~balance factor:  0.6
---Prediction...
~~~accuracy: 0.9788  <--train
---Start Validing...
~~~balance factor:  0.6
---Prediction...
~~~accuracy: 0.9778  <--valid
---
---Compute beta...
~~~balance factor:  0.7
---Prediction...
~~~accuracy: 0.9792  <--train
---Start Validing...
~~~balance factor:  0.7
---Prediction...
~~~accuracy: 0.9779  <--valid
---
---Compute beta...
~~~balance factor:  0.8
---Prediction...
~~~accuracy: 0.9793  <--train
---Start Validing...
~~~balance factor:  0.8
---Prediction...
~~~accuracy: 0.9781  <--valid
---
---Compute beta...
~~~balance factor:  0.9
---Prediction...
~~~accuracy: 0.9798  <--train
---Start Validing...
~~~balance factor:  0.9
---Prediction...
~~~accuracy: 0.9781  <--valid
---Done!
[Finished in 264.9s]
```




# 实验4

# 配置

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

        self._net['C1'] = tf.nn.relu(self._net['C1'])

        # self._net['P1'] = tf.nn.max_pool(
        self._net['P1'] = tt.nn.square_root_pool(
            self._net['C1'],
            ksize=[1, 3, 3, 1],  # [1, kH, kW, 1]
            strides=[1, 1, 1, 1],  # [1, sH, sW, 1]
            padding="VALID", data_format='NHWC', name='P1')

        self._net['P1'] = tf.nn.lrn(self._net['P1'])
        self._net['C2'] = tf.nn.conv2d(
            self._net['P1'], self._W['C2'],
            strides=[1, 1, 1, 1],  # [1, sH, sW, 1]
            padding="VALID", data_format='NHWC',
            dilations=[1, 1, 1, 1], name='C2')

        self._net['C2'] = tf.nn.relu(self._net['C2'])

        # self._net['P2'] = tf.nn.max_pool(
        self._net['P2'] = tt.nn.square_root_pool(
            self._net['C2'],
            ksize=[1, 3, 3, 1],  # [1, kH, kW, 1]
            strides=[1, 1, 1, 1],  # [1, sH, sW, 1]
            padding="VALID", data_format='NHWC', name='P2')

        self._net['P2'] = tf.nn.lrn(self._net['P2'])

        self._net['C3'] = tf.nn.conv2d(
            self._net['P2'], self._W['C3'],
            strides=[1, 1, 1, 1],  # [1, sH, sW, 1]
            padding="VALID", data_format='NHWC',
            dilations=[1, 1, 1, 1], name='C3')

        self._net['C3'] = tf.nn.relu(self._net['C3'])

        # self._net['P3'] = tf.nn.max_pool(
        self._net['P3'] = tt.nn.square_root_pool(
            self._net['C3'],
            ksize=[1, 3, 3, 1],  # [1, kH, kW, 1]
            strides=[1, 1, 1, 1],  # [1, sH, sW, 1]
            padding="VALID", data_format='NHWC', name='P3')

        self._net['P3'] = tf.nn.lrn(self._net['P3'])

        HO = self._net['P3']
        # HO = self._net['P1']
        # HO = tf.nn.relu(self._net['P1'])
        # HO = tf.nn.lrn(HO)
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
~~~accuracy: 0.9926  <--train
---Start Validing...
~~~balance factor:  100.0
---Prediction...
~~~accuracy: 0.9778  <--valid
---
---Compute beta...
~~~balance factor:  10.0
---Prediction...
~~~accuracy: 0.9869  <--train
---Start Validing...
~~~balance factor:  10.0
---Prediction...
~~~accuracy: 0.9793  <--valid
---
---Compute beta...
~~~balance factor:  5.0
---Prediction...
~~~accuracy: 0.9852  <--train
---Start Validing...
~~~balance factor:  5.0
---Prediction...
~~~accuracy: 0.9791  <--valid
---
---Compute beta...
~~~balance factor:  2.0
---Prediction...
~~~accuracy: 0.9823  <--train
---Start Validing...
~~~balance factor:  2.0
---Prediction...
~~~accuracy: 0.9794  <--valid
---
---Compute beta...
~~~balance factor:  1.5
---Prediction...
~~~accuracy: 0.9813  <--train
---Start Validing...
~~~balance factor:  1.5
---Prediction...
~~~accuracy: 0.9791  <--valid
---
---Compute beta...
~~~balance factor:  0.01
---Prediction...
~~~accuracy: 0.9476  <--train
---Start Validing...
~~~balance factor:  0.01
---Prediction...
~~~accuracy: 0.9503  <--valid
---
---Compute beta...
~~~balance factor:  0.1
---Prediction...
~~~accuracy: 0.9703  <--train
---Start Validing...
~~~balance factor:  0.1
---Prediction...
~~~accuracy: 0.9707  <--valid
---
---Compute beta...
~~~balance factor:  0.2
---Prediction...
~~~accuracy: 0.9737  <--train
---Start Validing...
~~~balance factor:  0.2
---Prediction...
~~~accuracy: 0.9743  <--valid
---
---Compute beta...
~~~balance factor:  0.3
---Prediction...
~~~accuracy: 0.9758  <--train
---Start Validing...
~~~balance factor:  0.3
---Prediction...
~~~accuracy: 0.9759  <--valid
---
---Compute beta...
~~~balance factor:  0.4
---Prediction...
~~~accuracy: 0.9769  <--train
---Start Validing...
~~~balance factor:  0.4
---Prediction...
~~~accuracy: 0.9767  <--valid
---
---Compute beta...
~~~balance factor:  0.5
---Prediction...
~~~accuracy: 0.9778  <--train
---Start Validing...
~~~balance factor:  0.5
---Prediction...
~~~accuracy: 0.9773  <--valid
---
---Compute beta...
~~~balance factor:  0.6
---Prediction...
~~~accuracy: 0.9783  <--train
---Start Validing...
~~~balance factor:  0.6
---Prediction...
~~~accuracy: 0.9777  <--valid
---
---Compute beta...
~~~balance factor:  0.7
---Prediction...
~~~accuracy: 0.9787  <--train
---Start Validing...
~~~balance factor:  0.7
---Prediction...
~~~accuracy: 0.9779  <--valid
---
---Compute beta...
~~~balance factor:  0.8
---Prediction...
~~~accuracy: 0.9792  <--train
---Start Validing...
~~~balance factor:  0.8
---Prediction...
~~~accuracy: 0.9784  <--valid
---
---Compute beta...
~~~balance factor:  0.9
---Prediction...
~~~accuracy: 0.9797  <--train
---Start Validing...
~~~balance factor:  0.9
---Prediction...
~~~accuracy: 0.9783  <--valid
---Done!
[Finished in 266.1s]
```


# 实验5

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

        self._net['C1'] = tf.nn.relu(self._net['C1'])

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

        self._net['C2'] = tf.nn.relu(self._net['C2'])

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

        self._net['C3'] = tf.nn.relu(self._net['C3'])

        # self._net['P3'] = tf.nn.max_pool(
        self._net['P3'] = tt.nn.square_root_pool(
            self._net['C3'],
            ksize=[1, 3, 3, 1],  # [1, kH, kW, 1]
            strides=[1, 1, 1, 1],  # [1, sH, sW, 1]
            padding="VALID", data_format='NHWC', name='P3')

        # self._net['P3'] = tf.nn.lrn(self._net['P3'])

        HO = self._net['P3']
        # HO = self._net['P1']
        # HO = tf.nn.relu(self._net['P1'])
        # HO = tf.nn.lrn(HO)
        HO = tf.nn.relu(HO)
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
~~~accuracy: 0.9944  <--train
---Start Validing...
~~~balance factor:  100.0
---Prediction...
~~~accuracy: 0.9779  <--valid
---
---Compute beta...
~~~balance factor:  10.0
---Prediction...
~~~accuracy: 0.9891  <--train
---Start Validing...
~~~balance factor:  10.0
---Prediction...
~~~accuracy: 0.9799  <--valid
---
---Compute beta...
~~~balance factor:  5.0
---Prediction...
~~~accuracy: 0.9875  <--train
---Start Validing...
~~~balance factor:  5.0
---Prediction...
~~~accuracy: 0.9802  <--valid
---
---Compute beta...
~~~balance factor:  2.0
---Prediction...
~~~accuracy: 0.9852  <--train
---Start Validing...
~~~balance factor:  2.0
---Prediction...
~~~accuracy: 0.9805  <--valid
---
---Compute beta...
~~~balance factor:  1.5
---Prediction...
~~~accuracy: 0.9840  <--train
---Start Validing...
~~~balance factor:  1.5
---Prediction...
~~~accuracy: 0.9804  <--valid
---
---Compute beta...
~~~balance factor:  0.01
---Prediction...
~~~accuracy: 0.9589  <--train
---Start Validing...
~~~balance factor:  0.01
---Prediction...
~~~accuracy: 0.9621  <--valid
---
---Compute beta...
~~~balance factor:  0.1
---Prediction...
~~~accuracy: 0.9754  <--train
---Start Validing...
~~~balance factor:  0.1
---Prediction...
~~~accuracy: 0.9765  <--valid
---
---Compute beta...
~~~balance factor:  0.2
---Prediction...
~~~accuracy: 0.9778  <--train
---Start Validing...
~~~balance factor:  0.2
---Prediction...
~~~accuracy: 0.9788  <--valid
---
---Compute beta...
~~~balance factor:  0.3
---Prediction...
~~~accuracy: 0.9790  <--train
---Start Validing...
~~~balance factor:  0.3
---Prediction...
~~~accuracy: 0.9792  <--valid
---
---Compute beta...
~~~balance factor:  0.4
---Prediction...
~~~accuracy: 0.9798  <--train
---Start Validing...
~~~balance factor:  0.4
---Prediction...
~~~accuracy: 0.9796  <--valid
---
---Compute beta...
~~~balance factor:  0.5
---Prediction...
~~~accuracy: 0.9804  <--train
---Start Validing...
~~~balance factor:  0.5
---Prediction...
~~~accuracy: 0.9797  <--valid
---
---Compute beta...
~~~balance factor:  0.6
---Prediction...
~~~accuracy: 0.9808  <--train
---Start Validing...
~~~balance factor:  0.6
---Prediction...
~~~accuracy: 0.9802  <--valid
---
---Compute beta...
~~~balance factor:  0.7
---Prediction...
~~~accuracy: 0.9816  <--train
---Start Validing...
~~~balance factor:  0.7
---Prediction...
~~~accuracy: 0.9803  <--valid
---
---Compute beta...
~~~balance factor:  0.8
---Prediction...
~~~accuracy: 0.9820  <--train
---Start Validing...
~~~balance factor:  0.8
---Prediction...
~~~accuracy: 0.9803  <--valid
---
---Compute beta...
~~~balance factor:  0.9
---Prediction...
~~~accuracy: 0.9822  <--train
---Start Validing...
~~~balance factor:  0.9
---Prediction...
~~~accuracy: 0.9803  <--valid
---Done!
[Finished in 265.7s]
```


# 实验6


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

        HO = self._net['P3']
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
~~~accuracy: 0.9892  <--train
---Start Validing...
~~~balance factor:  100.0
---Prediction...
~~~accuracy: 0.9833  <--valid
---
---Compute beta...
~~~balance factor:  10.0
---Prediction...
~~~accuracy: 0.9834  <--train
---Start Validing...
~~~balance factor:  10.0
---Prediction...
~~~accuracy: 0.9824  <--valid
---
---Compute beta...
~~~balance factor:  5.0
---Prediction...
~~~accuracy: 0.9808  <--train
---Start Validing...
~~~balance factor:  5.0
---Prediction...
~~~accuracy: 0.9808  <--valid
---
---Compute beta...
~~~balance factor:  2.0
---Prediction...
~~~accuracy: 0.9770  <--train
---Start Validing...
~~~balance factor:  2.0
---Prediction...
~~~accuracy: 0.9776  <--valid
---
---Compute beta...
~~~balance factor:  1.5
---Prediction...
~~~accuracy: 0.9752  <--train
---Start Validing...
~~~balance factor:  1.5
---Prediction...
~~~accuracy: 0.9759  <--valid
---
---Compute beta...
~~~balance factor:  0.01
---Prediction...
~~~accuracy: 0.8904  <--train
---Start Validing...
~~~balance factor:  0.01
---Prediction...
~~~accuracy: 0.8992  <--valid
---
---Compute beta...
~~~balance factor:  0.1
---Prediction...
~~~accuracy: 0.9494  <--train
---Start Validing...
~~~balance factor:  0.1
---Prediction...
~~~accuracy: 0.9541  <--valid
---
---Compute beta...
~~~balance factor:  0.2
---Prediction...
~~~accuracy: 0.9588  <--train
---Start Validing...
~~~balance factor:  0.2
---Prediction...
~~~accuracy: 0.9616  <--valid
---
---Compute beta...
~~~balance factor:  0.3
---Prediction...
~~~accuracy: 0.9631  <--train
---Start Validing...
~~~balance factor:  0.3
---Prediction...
~~~accuracy: 0.9660  <--valid
---
---Compute beta...
~~~balance factor:  0.4
---Prediction...
~~~accuracy: 0.9660  <--train
---Start Validing...
~~~balance factor:  0.4
---Prediction...
~~~accuracy: 0.9678  <--valid
---
---Compute beta...
~~~balance factor:  0.5
---Prediction...
~~~accuracy: 0.9677  <--train
---Start Validing...
~~~balance factor:  0.5
---Prediction...
~~~accuracy: 0.9696  <--valid
---
---Compute beta...
~~~balance factor:  0.6
---Prediction...
~~~accuracy: 0.9688  <--train
---Start Validing...
~~~balance factor:  0.6
---Prediction...
~~~accuracy: 0.9700  <--valid
---
---Compute beta...
~~~balance factor:  0.7
---Prediction...
~~~accuracy: 0.9701  <--train
---Start Validing...
~~~balance factor:  0.7
---Prediction...
~~~accuracy: 0.9710  <--valid
---
---Compute beta...
~~~balance factor:  0.8
---Prediction...
~~~accuracy: 0.9711  <--train
---Start Validing...
~~~balance factor:  0.8
---Prediction...
~~~accuracy: 0.9718  <--valid
---
---Compute beta...
~~~balance factor:  0.9
---Prediction...
~~~accuracy: 0.9720  <--train
---Start Validing...
~~~balance factor:  0.9
---Prediction...
~~~accuracy: 0.9725  <--valid
---Done!
[Finished in 269.4s]
```
