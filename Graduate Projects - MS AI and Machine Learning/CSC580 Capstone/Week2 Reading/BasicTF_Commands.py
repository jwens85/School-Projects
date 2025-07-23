import tensorflow as tf
import numpy as np

print("\n" + "="*60)
print("TENSORFLOW 2.X TENSOR BASICS DEMO")
print("="*60)

# Basic zero and one tensors
print("tf.zeros(2):\n", tf.zeros(2).numpy())
print("tf.zeros((2, 3)):\n", tf.zeros((2, 3)).numpy())
print("tf.ones((2, 2, 2)):\n", tf.ones((2, 2, 2)).numpy())

# Fill and constant
print("tf.fill((2, 2), 5.0):\n", tf.fill((2, 2), value=5.0).numpy())
print("tf.constant(3):\n", tf.constant(3).numpy())

# Random tensors
print("tf.random.normal:\n", tf.random.normal((2, 2), mean=0, stddev=1).numpy())
print("tf.random.uniform:\n", tf.random.uniform((2, 2), minval=-2, maxval=2).numpy())

# Tensor arithmetic
c = tf.ones((2, 2))
d = tf.ones((2, 2))
e = c + d
f = 2 * e
print("Elementwise addition:\n", e.numpy())
print("Scalar multiplication:\n", f.numpy())

# Elementwise multiplication
c = tf.fill((2, 2), 2.0)
d = tf.fill((2, 2), 7.0)
e = c * d
print("Elementwise multiplication:\n", e.numpy())

# Identity matrix and diagonal matrix
print("Identity matrix:\n", tf.eye(4).numpy())
r = tf.range(1, 5, 1)
print("Range (1 to 4):\n", r.numpy())
print("Diagonal matrix:\n", tf.linalg.diag(r).numpy())

# Transpose
a = tf.ones((2, 3))
print("Transpose:\n", tf.transpose(a).numpy())

# Matrix multiplication
a = tf.ones((2, 3))
b = tf.ones((3, 4))
print("Matrix multiplication:\n", tf.matmul(a, b).numpy())

# Type casting
a = tf.ones((2, 2), dtype=tf.int32)
b = tf.cast(a, tf.float32)
print("Original int32:\n", a.numpy())
print("Casted to float32:\n", b.numpy())

# Reshape
a = tf.ones(8)
print("Reshape to (4,2):\n", tf.reshape(a, (4, 2)).numpy())
print("Reshape to (2,2,2):\n", tf.reshape(a, (2, 2, 2)).numpy())

# Shapes and expand/squeeze
a = tf.ones(2)
print("Shape of a:", a.shape)
b = tf.expand_dims(a, 0)
print("Expand dims (0):", b.shape, b.numpy())
c = tf.expand_dims(a, 1)
print("Expand dims (1):", c.shape, c.numpy())
d = tf.squeeze(b)
print("Squeezed shape:", d.shape, d.numpy())

# Broadcasting
a = tf.ones((2, 2))
b = tf.range(0, 2, 1, dtype=tf.float32)
c = a + b  # Broadcasting
print("Broadcast result:\n", c.numpy())

print("="*60)
print("TensorFlow 2.x Tensor Demo Completed")
print("="*60)
