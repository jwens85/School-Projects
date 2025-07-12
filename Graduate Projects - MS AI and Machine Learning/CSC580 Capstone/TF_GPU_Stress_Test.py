import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import time
import numpy as np

print("=" * 60)
print("TENSORFLOW GPU STRESS TEST")
print("=" * 60)

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU Available: {gpus[0].name}")
    print(f"GPU Memory: {tf.config.experimental.get_memory_info('GPU:0')['peak'] / 1024 ** 3:.2f} GB used")
else:
    print("No GPU detected - running on CPU")

print("=" * 60)

# Large matrix operations
print("Running large matrix multiplication stress test...")
matrix_size = 4000
iterations = 10

with tf.device('/GPU:0' if gpus else '/CPU:0'):
    # Create large random matrices
    A = tf.random.normal([matrix_size, matrix_size], dtype=tf.float32)
    B = tf.random.normal([matrix_size, matrix_size], dtype=tf.float32)

    print(f"Matrix dimensions: {matrix_size} x {matrix_size}")
    print(f"Operations: {iterations} matrix multiplications")

    start_time = time.time()

    for i in range(iterations):
        C = tf.matmul(A, B)
        if i % 2 == 0:
            print(f"Iteration {i + 1}/{iterations} completed")

    end_time = time.time()

    print(f"Total computation time: {end_time - start_time:.2f} seconds")
    print(f"Average time per operation: {(end_time - start_time) / iterations:.3f} seconds")

print("=" * 60)

# Convolution operations (typical for neural networks)
print("Running convolution operations...")

with tf.device('/GPU:0' if gpus else '/CPU:0'):
    # Create a batch of images
    batch_size = 32
    image_size = 512
    channels = 3
    filters = 64

    images = tf.random.normal([batch_size, image_size, image_size, channels])
    kernel = tf.random.normal([5, 5, channels, filters])

    print(f"Image batch: {batch_size} images of {image_size}x{image_size}x{channels}")
    print(f"Convolution filters: {filters}")

    start_time = time.time()

    for i in range(5):
        conv_result = tf.nn.conv2d(images, kernel, strides=[1, 1, 1, 1], padding='SAME')
        pooled = tf.nn.max_pool2d(conv_result, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print(f"Convolution {i + 1}/5 completed - Output shape: {pooled.shape}")

    end_time = time.time()

    print(f"Convolution operations completed in: {end_time - start_time:.2f} seconds")

print("=" * 60)

# Neural network training simulation
print("Simulating neural network training...")

with tf.device('/GPU:0' if gpus else '/CPU:0'):
    # Create a simple but computationally intensive model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1024, activation='relu', input_shape=(512,)),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Generate synthetic training data
    x_train = tf.random.normal([5000, 512])
    y_train = tf.keras.utils.to_categorical(np.random.randint(0, 10, 5000), 10)

    print(f"Model parameters: {model.count_params():,}")
    print("Training for 3 epochs...")

    start_time = time.time()
    history = model.fit(x_train, y_train, epochs=3, batch_size=64, verbose=1)
    end_time = time.time()

    print(f"Training completed in: {end_time - start_time:.2f} seconds")
    print(f"Final loss: {history.history['loss'][-1]:.4f}")
    print(f"Final accuracy: {history.history['accuracy'][-1]:.4f}")

print("=" * 60)
print("GPU STRESS TEST COMPLETED SUCCESSFULLY")
print("=" * 60)

# Final GPU memory check
if gpus:
    final_memory = tf.config.experimental.get_memory_info('GPU:0')['peak'] / 1024 ** 3
    print(f"Peak GPU memory usage: {final_memory:.2f} GB")