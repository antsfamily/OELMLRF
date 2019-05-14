import tensorflow as tf

tf.set_random_seed(None)
a = tf.random_uniform([1])
b = tf.random_normal([1])

# Repeatedly running this block with the same graph will generate the same
# sequences of 'a' and 'b'.
print("Session 1")

H = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='factor')
N, L = H.get_shape()


if N > 1:
    print("===============")
    y = tf.eye(3) / 0.2
else:
    print("---------------")
    y = tf.eye(3) / 0.5

with tf.Session() as sess1:
    print(sess1.run(a))  # generates 'A1'
    print(sess1.run(a))  # generates 'A2'
    print(sess1.run(b))  # generates 'B1'
    print(sess1.run(b))  # generates 'B2'

    print(sess1.run(y, {H: [[1, 2]]}))
    print(sess1.run(y, {H: [[1, 2], [3, 4], [5, 6]]}))

print("Session 2")
with tf.Session() as sess2:
    print(sess2.run(a))  # generates 'A1'
    print(sess2.run(a))  # generates 'A2'
    print(sess2.run(b))  # generates 'B1'
    print(sess2.run(b))  # generates 'B2'
