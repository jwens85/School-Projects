import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# Force GPU initialization early and quietly
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Pre-initialize GPU to get the device creation message out of the way
    with tf.device('/GPU:0'):
        _ = tf.constant([1.0])  # Simple operation to trigger GPU initialization

print("\n" + "="*60)
print("TENSORFLOW GPU TEST RESULTS")
print("="*60)

gpu_available = len(gpus) > 0

print(f"GPU Available: {gpu_available}")
if gpu_available:
    print(f"Using GPU for computation")
else:
    print(f"Using CPU for computation")

print(f"\nRunning matrix multiplication test...")

with tf.device('/GPU:0' if gpu_available else '/CPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 0.0], [0.0, 1.0]])
    c = tf.matmul(a, b)

print(f"Result:")
print(c.numpy())
print(f"\n✨ Test completed successfully!")
print("="*60)