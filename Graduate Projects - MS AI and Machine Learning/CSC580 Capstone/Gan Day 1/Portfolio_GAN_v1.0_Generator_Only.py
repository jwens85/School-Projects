import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import (
    Input, Dense, Reshape, Flatten,
    BatchNormalization, LeakyReLU,
    Conv2D, Conv2DTranspose, Dropout
)
from keras.optimizers import Adam
import tensorflow as tf

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# Parameters
image_shape = (32, 32, 3)
latent_dim = 100

# Output directory
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
base_output_dir = os.path.join("outputs", "v1.0_generator_only", timestamp)
os.makedirs(base_output_dir, exist_ok=True)

# Load CIFAR-10 ships
(X, y), (_, _) = cifar10.load_data()
X = X[y.flatten() == 8]  # Ships only
print(f"Loaded {len(X)} ship images")

# Normalize to [-1, 1] for tanh
X_train = (X.astype(np.float32) - 127.5) / 127.5
print(f"Data range: [{X_train.min():.2f}, {X_train.max():.2f}]")


def make_generator():
    """Generator architecture - let's test this in isolation"""
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
    """Create an encoder to map real images to latent space"""
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
print("Building Generator and Encoder for testing...")
generator = make_generator()
encoder = make_encoder()

print("\nGenerator:")
generator.summary()
print("\nEncoder:")
encoder.summary()

# Create autoencoder (encoder -> generator)
autoencoder_input = Input(shape=image_shape)
encoded = encoder(autoencoder_input)
decoded = generator(encoded)
autoencoder = Model(autoencoder_input, decoded)

# Compile autoencoder with simple reconstruction loss
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

print("\nAutoencoder (for testing generator):")
autoencoder.summary()

# Test 1: Train as autoencoder to see if generator can produce meaningful images
print("\n" + "=" * 60)
print("TEST 1: AUTOENCODER TRAINING")
print("Testing if generator can learn to reconstruct real images")
print("=" * 60)

EPOCHS_AE = 500
BATCH_SIZE = 32

# Train autoencoder
print("Training autoencoder...")
for epoch in range(EPOCHS_AE):
    # Get random batch
    idx = np.random.randint(0, X_train.shape[0], BATCH_SIZE)
    real_images = X_train[idx]

    # Train autoencoder (real -> encoded -> reconstructed)
    loss = autoencoder.train_on_batch(real_images, real_images)

    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d}: Reconstruction loss = {loss:.4f}")

    if epoch % 100 == 0:
        # Test reconstruction
        test_images = X_train[:8]
        reconstructed = autoencoder.predict(test_images, verbose=0)

        # Display original vs reconstructed
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))

        # Original images
        for i in range(8):
            orig_img = 0.5 * test_images[i] + 0.5
            axes[0, i].imshow(orig_img)
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')

        # Reconstructed images
        for i in range(8):
            recon_img = 0.5 * reconstructed[i] + 0.5
            recon_img = np.clip(recon_img, 0, 1)
            axes[1, i].imshow(recon_img)
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')

        plt.suptitle(f'Autoencoder Test - Epoch {epoch}')
        plt.tight_layout()
        plt.savefig(os.path.join(base_output_dir, f"autoencoder_epoch_{epoch}.png"),
                    bbox_inches='tight', dpi=100)
        plt.close()

print("\nAutoencoder training completed!")

# Test 2: Generator with random noise
print("\n" + "=" * 60)
print("TEST 2: GENERATOR WITH RANDOM NOISE")
print("Testing if trained generator can create images from random noise")
print("=" * 60)


def test_generator_with_noise(epoch_name):
    """Test generator with random noise"""
    # Generate from random noise
    noise = np.random.normal(0, 1, (16, latent_dim))
    generated = generator.predict(noise, verbose=0)

    # Display generated images
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(16):
        row, col = i // 4, i % 4
        gen_img = 0.5 * generated[i] + 0.5
        gen_img = np.clip(gen_img, 0, 1)
        axes[row, col].imshow(gen_img)
        axes[row, col].axis('off')

    plt.suptitle(f'Generator from Random Noise - {epoch_name}')
    plt.tight_layout()
    filename = f"generator_noise_{epoch_name.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(base_output_dir, filename), bbox_inches='tight', dpi=100)
    plt.close()
    return filename


# Test generator before any training
print("Testing generator before training...")
test_generator_with_noise("Before Training")

# Test generator after autoencoder training
print("Testing generator after autoencoder training...")
test_generator_with_noise("After Autoencoder")

# Test 3: Train generator directly with "fake" target images
print("\n" + "=" * 60)
print("TEST 3: DIRECT GENERATOR TRAINING")
print("Training generator to match averaged ship images")
print("=" * 60)

# Create "average" ship image as target
avg_ship = np.mean(X_train, axis=0)
print(f"Average ship shape: {avg_ship.shape}")

# Train generator to produce the average ship from any noise
generator.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

EPOCHS_DIRECT = 200
print("Training generator directly...")

for epoch in range(EPOCHS_DIRECT):
    # Generate random noise
    noise = np.random.normal(0, 1, (BATCH_SIZE, latent_dim))

    # Target: average ship repeated for batch
    targets = np.tile(avg_ship, (BATCH_SIZE, 1, 1, 1))

    # Train generator to produce average ship
    loss = generator.train_on_batch(noise, targets)

    if epoch % 25 == 0:
        print(f"Epoch {epoch:3d}: Direct generator loss = {loss:.4f}")

    if epoch % 50 == 0:
        test_generator_with_noise(f"Direct Training Epoch {epoch}")

print("Direct generator training completed!")

# Test 4: Final comprehensive test
print("\n" + "=" * 60)
print("TEST 4: FINAL GENERATOR EVALUATION")
print("=" * 60)

# Show what the generator learned
fig, axes = plt.subplots(3, 8, figsize=(16, 6))

# Row 1: Real ships
real_sample = X_train[:8]
for i in range(8):
    img = 0.5 * real_sample[i] + 0.5
    axes[0, i].imshow(img)
    axes[0, i].set_title('Real Ship')
    axes[0, i].axis('off')

# Row 2: Reconstructed (via autoencoder)
reconstructed = autoencoder.predict(real_sample, verbose=0)
for i in range(8):
    img = 0.5 * reconstructed[i] + 0.5
    img = np.clip(img, 0, 1)
    axes[1, i].imshow(img)
    axes[1, i].set_title('Reconstructed')
    axes[1, i].axis('off')

# Row 3: Generated from noise
noise = np.random.normal(0, 1, (8, latent_dim))
generated = generator.predict(noise, verbose=0)
for i in range(8):
    img = 0.5 * generated[i] + 0.5
    img = np.clip(img, 0, 1)
    axes[2, i].imshow(img)
    axes[2, i].set_title('From Noise')
    axes[2, i].axis('off')

plt.suptitle('Final Generator Evaluation')
plt.tight_layout()
plt.savefig(os.path.join(base_output_dir, "final_generator_evaluation.png"),
            bbox_inches='tight', dpi=100)
plt.close()

# Save the trained generator
generator.save(os.path.join(base_output_dir, "trained_generator.h5"))

print("\n" + "=" * 60)
print("GENERATOR TESTING COMPLETE")
print("=" * 60)
print("Check the generated images to see:")
print("1. Can the generator reconstruct real images? (autoencoder test)")
print("2. Can the generator create anything meaningful from random noise?")
print("3. Did direct training help the generator learn ship-like features?")
print("")
print("If the generator works here, we know the architecture is good")
print("If it doesn't work, we need to fix the generator architecture")
print("If it works, then the problem is in the adversarial training setup")
print(f"\nAll results saved to: {base_output_dir}")
print("=" * 60)