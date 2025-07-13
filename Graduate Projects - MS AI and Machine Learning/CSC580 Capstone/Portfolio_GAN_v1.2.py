import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import (
    Input, Dense, Reshape, Flatten,
    BatchNormalization, LeakyReLU,
    Conv2D, Conv2DTranspose
)
from keras.optimizers import Adam
import tensorflow as tf

# Set seeds for reproducible training, but we'll vary the display
np.random.seed(42)
tf.random.set_seed(42)

# Parameters
image_shape = (32, 32, 3)
latent_dim = 100

# Timestamped output directory - matching your style
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
base_output_dir = os.path.join("outputs", "v1.2", timestamp)
os.makedirs(base_output_dir, exist_ok=True)

# Load CIFAR-10 ships
(X, y), (_, _) = cifar10.load_data()
X = X[y.flatten() == 8]
print(f"Loaded {len(X)} ship images")

# Normalize to [-1, 1] for tanh
X_train = (X.astype(np.float32) - 127.5) / 127.5
print(f"Data range: [{X_train.min():.2f}, {X_train.max():.2f}]")


def make_generator():
    """Generator - same proven architecture"""
    model = Sequential()

    # Project and reshape
    model.add(Dense(8 * 8 * 256, input_dim=latent_dim))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Reshape((8, 8, 256)))

    # Upsample to 16x16
    model.add(Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    # Upsample to 32x32
    model.add(Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', activation='tanh'))

    return model


def make_encoder():
    """Encoder to pair with generator"""
    model = Sequential()

    # Downsample to 16x16
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=image_shape))
    model.add(LeakyReLU(0.2))

    # Downsample to 8x8
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))

    # Map to latent space
    model.add(Flatten())
    model.add(Dense(latent_dim))

    return model


# Build models
print("Building Generator and Encoder...")
generator = make_generator()
encoder = make_encoder()

# Create autoencoder
autoencoder_input = Input(shape=image_shape)
encoded = encoder(autoencoder_input)
decoded = generator(encoded)
autoencoder = Model(autoencoder_input, decoded)

# Compile autoencoder
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

print("Starting autoencoder training...")
print(f"Output directory: {base_output_dir}")

# Training parameters
EPOCHS = 100001
BATCH_SIZE = 32
SAVE_INTERVAL = 10000

# Training loop
for epoch in range(EPOCHS):
    # Get random batch from ALL ships for training
    idx = np.random.randint(0, X_train.shape[0], BATCH_SIZE)
    real_images = X_train[idx]

    # Train autoencoder
    loss = autoencoder.train_on_batch(real_images, real_images)

    # Print progress
    if epoch % 50 == 0:
        print(f"Epoch {epoch:4d}: Reconstruction loss = {loss:.6f}")

    # Save reconstruction results
    if epoch % SAVE_INTERVAL == 0:
        # NEW: Select 8 RANDOM ships for testing each time
        # This will show the generator works on the full dataset variety
        random_indices = np.random.choice(X_train.shape[0], 8, replace=False)
        test_images = X_train[random_indices]
        reconstructed = autoencoder.predict(test_images, verbose=0)

        # Create comparison plot
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))

        # Original images (top row) - now showing random ships
        for i in range(8):
            orig_img = 0.5 * test_images[i] + 0.5
            orig_img = np.clip(orig_img, 0, 1)
            axes[0, i].imshow(orig_img)
            axes[0, i].set_title(f'Original #{random_indices[i]}')
            axes[0, i].axis('off')

        # Reconstructed images (bottom row)
        for i in range(8):
            recon_img = 0.5 * reconstructed[i] + 0.5
            recon_img = np.clip(recon_img, 0, 1)
            axes[1, i].imshow(recon_img)
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')

        plt.suptitle(f'Random Ships Test - Epoch {epoch} (Loss: {loss:.6f})')
        plt.tight_layout()

        # Save with consistent naming
        filename = f"autoencoder_epoch_{epoch}.png"
        filepath = os.path.join(base_output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', dpi=100)
        plt.close()

        print(f"Saved: {filename} (showing ships: {random_indices})")

print("\nAutoencoder training completed!")

# Save final model
generator.save(os.path.join(base_output_dir, "generator_trained.h5"))
encoder.save(os.path.join(base_output_dir, "encoder_trained.h5"))

print("\nTraining Summary:")
print(f"- Total epochs: {EPOCHS}")
print(f"- Final reconstruction loss: {loss:.6f}")
print(f"- Models saved to: {base_output_dir}")
print(f"- Generated {(EPOCHS // SAVE_INTERVAL) + 1} autoencoder_epoch_xxx.png files")
print(f"- Each epoch shows 8 random ships from the full {len(X_train)} ship dataset")

print("\nNext steps:")
print("1. Check that reconstruction works well on DIVERSE ships")
print("2. Verify generator learned general ship features, not just specific ones")
print("3. Ready to move to GAN training with random noise input!")