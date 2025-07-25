import os
import sys
import warnings

# Set environment variables before TensorFlow is imported
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN messages  
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')  # Keep GPU enabled if available

# Additional suppression flags
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'

# Suppress Python warnings
warnings.filterwarnings('ignore')

# Try to suppress absl logging if it's available
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except ImportError:
    pass

# Function to configure TensorFlow after it's imported
def configure_tensorflow():
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        tf.autograph.set_verbosity(0)
        tf.config.set_soft_device_placement(True)
        
        # Optionally verify GPU is available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"TensorFlow configured with {len(gpus)} GPU(s) available")
        else:
            print("TensorFlow configured (CPU only)")
            
    except ImportError:
        print("TensorFlow not installed. Please install it first.")
    except Exception as e:
        print(f"Error configuring TensorFlow: {e}")

# Autoconfigure if TensorFlow is already imported
if 'tensorflow' in sys.modules:
    configure_tensorflow()

print("TF Warnings Suppressor loaded. Most TensorFlow warnings will be suppressed.")
print("Note: Some C++ backend warnings may still appear in red.")