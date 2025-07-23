import os
import sys
import tensorflow as tf

# Suppress TensorFlow log messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Force early GPU initialization
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        with tf.device('/GPU:0'):
            _ = tf.constant([1.0])
    except RuntimeError as e:
        print(f"RuntimeError during GPU warm-up: {e}")

print("\n" + "="*60)
print("TENSORFLOW GPU TEST RESULTS")
print("="*60)

# TensorFlow version
print(f"TensorFlow Version: {tf.__version__}")

# Python version
print(f"Python Version: {sys.version.split()[0]}")

# CUDA and cuDNN support (inferred)
build_info = tf.sysconfig.get_build_info()
cuda_version = build_info.get('cuda_version', 'Unknown')
cudnn_version = build_info.get('cudnn_version', 'Unknown')

print(f"CUDA Version (from build): {cuda_version}")
print(f"cuDNN Version (from build): {cudnn_version}")

# GPU status
gpu_available = len(gpus) > 0
print(f"\nGPU Available: {gpu_available}")

if gpu_available:
    for idx, gpu in enumerate(gpus):
        print(f"GPU {idx}: {gpu.name}")
else:
    print("No GPU devices detected. Defaulting to CPU.")

# Matrix multiplication test
print(f"\nRunning matrix multiplication test...")
with tf.device('/GPU:0' if gpu_available else '/CPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 0.0], [0.0, 1.0]])
    c = tf.matmul(a, b)

print(f"Result:")
print(c.numpy())
print("\nTest completed successfully.")
print("="*60)