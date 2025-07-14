#!/usr/bin/env python3
"""
TensorFlow/Keras Installation Test Script
This script performs comprehensive tests to verify TensorFlow and Keras are working correctly.
"""

import sys
import os


def test_imports():
    """Test basic imports of TensorFlow and Keras"""
    print("=" * 50)
    print("TESTING IMPORTS")
    print("=" * 50)

    try:
        import tensorflow as tf
        print(f"✓ TensorFlow imported successfully")
        print(f"✓ TensorFlow version: {tf.__version__}")

        from tensorflow import keras
        print(f"✓ Keras imported successfully")
        print(f"✓ Keras version: {keras.__version__}")

        return tf, keras
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return None, None


def test_gpu_availability(tf):
    """Test GPU availability and configuration"""
    print("\n" + "=" * 50)
    print("TESTING GPU AVAILABILITY")
    print("=" * 50)

    if tf is None:
        print("✗ Cannot test GPU - TensorFlow not available")
        return

    # Check if GPU is available
    gpu_available = tf.config.list_physical_devices('GPU')
    print(f"GPU devices found: {len(gpu_available)}")

    if gpu_available:
        print("✓ GPU is available")
        for i, gpu in enumerate(gpu_available):
            print(f"  GPU {i}: {gpu}")

        # Test GPU memory growth (prevents TensorFlow from allocating all GPU memory)
        try:
            for gpu in gpu_available:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✓ GPU memory growth configured")
        except Exception as e:
            print(f"⚠ GPU memory growth configuration failed: {e}")
    else:
        print("⚠ No GPU detected - will use CPU")


def test_basic_operations(tf):
    """Test basic TensorFlow operations"""
    print("\n" + "=" * 50)
    print("TESTING BASIC OPERATIONS")
    print("=" * 50)

    if tf is None:
        print("✗ Cannot test operations - TensorFlow not available")
        return

    try:
        # Test basic tensor operations
        a = tf.constant([1, 2, 3, 4])
        b = tf.constant([5, 6, 7, 8])
        c = tf.add(a, b)
        print(f"✓ Basic tensor operations work")
        print(f"  a = {a.numpy()}")
        print(f"  b = {b.numpy()}")
        print(f"  a + b = {c.numpy()}")

        # Test matrix operations
        matrix1 = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
        matrix2 = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
        result = tf.matmul(matrix1, matrix2)
        print(f"✓ Matrix operations work")
        print(f"  Matrix multiplication result:\n{result.numpy()}")

    except Exception as e:
        print(f"✗ Basic operations failed: {e}")


def test_keras_model(tf, keras):
    """Test creating and training a simple Keras model"""
    print("\n" + "=" * 50)
    print("TESTING KERAS MODEL")
    print("=" * 50)

    if tf is None or keras is None:
        print("✗ Cannot test model - TensorFlow/Keras not available")
        return

    try:
        # Generate simple test data
        import numpy as np
        X = np.random.random((100, 10))
        y = np.random.randint(2, size=(100, 1))

        # Create a simple model
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(10,)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        print("✓ Model created successfully")
        print(f"  Model has {model.count_params()} parameters")

        # Compile the model
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        print("✓ Model compiled successfully")

        # Train for a few epochs
        print("  Training model for 3 epochs...")
        history = model.fit(X, y, epochs=3, batch_size=32, verbose=0)

        final_loss = history.history['loss'][-1]
        final_accuracy = history.history['accuracy'][-1]

        print(f"✓ Model trained successfully")
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Final accuracy: {final_accuracy:.4f}")

        # Test prediction
        test_input = np.random.random((1, 10))
        prediction = model.predict(test_input, verbose=0)
        print(f"✓ Model prediction works: {prediction[0][0]:.4f}")

    except Exception as e:
        print(f"✗ Keras model test failed: {e}")


def test_data_pipeline(tf):
    """Test TensorFlow data pipeline"""
    print("\n" + "=" * 50)
    print("TESTING DATA PIPELINE")
    print("=" * 50)

    if tf is None:
        print("✗ Cannot test data pipeline - TensorFlow not available")
        return

    try:
        # Create a simple dataset
        dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
        dataset = dataset.batch(2)

        print("✓ Dataset created successfully")

        # Test iteration
        for batch in dataset:
            print(f"  Batch: {batch.numpy()}")

        print("✓ Data pipeline works correctly")

    except Exception as e:
        print(f"✗ Data pipeline test failed: {e}")


def print_system_info():
    """Print system information"""
    print("\n" + "=" * 50)
    print("SYSTEM INFORMATION")
    print("=" * 50)

    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")

    try:
        import tensorflow as tf
        print(f"TensorFlow build info:")
        print(f"  Built with CUDA: {tf.test.is_built_with_cuda()}")
        print(f"  Built with GPU support: {tf.test.is_built_with_gpu_support()}")
    except:
        pass


def main():
    """Run all tests"""
    print("TensorFlow/Keras Installation Test")
    print("This script will test your TensorFlow and Keras installation")
    print("Run this in your PyCharm terminal or Ubuntu terminal")

    # Print system info
    print_system_info()

    # Test imports
    tf, keras = test_imports()

    # Test GPU availability
    test_gpu_availability(tf)

    # Test basic operations
    test_basic_operations(tf)

    # Test Keras model
    test_keras_model(tf, keras)

    # Test data pipeline
    test_data_pipeline(tf)

    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    if tf is not None and keras is not None:
        print("✓ TensorFlow and Keras appear to be working correctly!")
        print("✓ You should be ready to start your ML projects")
    else:
        print("✗ There are issues with your TensorFlow/Keras installation")
        print("  Try reinstalling with: pip install tensorflow")
        print("  Or for GPU support: pip install tensorflow-gpu")

    print("\nIf you see any errors above, please share them for troubleshooting.")


if __name__ == "__main__":
    main()