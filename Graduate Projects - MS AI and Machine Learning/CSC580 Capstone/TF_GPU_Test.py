import tensorflow as tf

print("Available devices:", tf.config.list_physical_devices())

with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 0.0], [0.0, 1.0]])
    c = tf.matmul(a, b)

print("Result:", c.numpy())
