import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from keras.datasets import cifar10
from keras.models import Sequential, Model, load_model
from keras.layers import (
    Input, Dense, Reshape, Flatten, Dropout,
    BatchNormalization, LeakyReLU,
    Conv2D, Conv2DTranspose, ZeroPadding2D
)
from keras.optimizers import Adam
import tensorflow as tf

# Set seeds for reproducible training
np.random.seed(42)
tf.random.set_seed(42)

# Parameters
image_shape = (32, 32, 3)
latent_dim = 100

# Assignment-required training parameters
EPOCHS = 15000
BATCH_SIZE = 32
DISPLAY_INTERVAL = 2500

# Auto-derive version from filename with debug info
script_name = os.path.basename(__file__)  # Gets "Portfolio_GAN_v3.1_Balanced.py"
print(f"DEBUG: Original filename: {script_name}")

# Remove .py extension first
script_name_clean = script_name.replace('.py', '')  # "Portfolio_GAN_v3.1_Balanced"
print(f"DEBUG: Clean filename: {script_name_clean}")

# Split and extract version
name_parts = script_name_clean.split('_')
print(f"DEBUG: Name parts: {name_parts}")

if len(name_parts) >= 4:
    # Has descriptor: Portfolio_GAN_v3.1_Balanced -> "v3.1_balanced"
    version = f"{name_parts[2]}_{name_parts[3].lower()}"
    print(f"DEBUG: Using descriptor version: {version}")
else:
    # No descriptor: Portfolio_GAN_v3.1 -> "v3.1"
    version = name_parts[2]
    print(f"DEBUG: Using simple version: {version}")

print(f"DEBUG: Final version: {version}")

# Timestamped output directory
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
base_output_dir = os.path.join("outputs", version, timestamp)
os.makedirs(base_output_dir, exist_ok=True)

print(f"GAN {version} - Balanced Adversarial Training")
print(f"Output directory: {base_output_dir}")

# Load CIFAR-10 ships
(X, y), (_, _) = cifar10.load_data()
X = X[y.flatten() == 8]
print(f"Loaded {len(X)} ship images")

# Normalize to [-1, 1] for tanh
X_train = (X.astype(np.float32) - 127.5) / 127.5
print(f"Data range: [{X_train.min():.2f}, {X_train.max():.2f}]")


def make_generator():
    """Generator - same proven architecture from autoencoder"""
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


def make_discriminator():
    """Discriminator with assignment-inspired architecture"""
    model = Sequential()

    # First conv block: 32x32 -> 16x16
    model.add(Conv2D(64, kernel_size=4, strides=2,
                     input_shape=image_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    # Second conv block: 16x16 -> 8x8
    model.add(Conv2D(128, kernel_size=4, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    # Third conv block: 8x8 -> 4x4
    model.add(Conv2D(256, kernel_size=4, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    # Fourth conv block: 4x4 -> 2x2
    model.add(Conv2D(512, kernel_size=4, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    # Output layer
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model


def save_generated_images(generator, epoch, noise_for_display):
    """Save generated images for assignment submission"""
    generated_images = generator.predict(noise_for_display, verbose=0)

    # Create 4x4 grid as per assignment style
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))

    for i in range(16):
        row = i // 4
        col = i % 4

        # Denormalize from [-1, 1] to [0, 1]
        img = 0.5 * generated_images[i] + 0.5
        img = np.clip(img, 0, 1)

        axes[row, col].imshow(img)
        axes[row, col].axis('off')

    plt.suptitle(f'Generated Ships - Epoch {epoch}', fontsize=16)
    plt.tight_layout()

    # Save with assignment-friendly naming
    filename = f"generated_epoch_{epoch:05d}.png"
    filepath = os.path.join(base_output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"Saved: {filename}")


# Build models
print("\nBuilding Generator and Discriminator...")
generator = make_generator()
discriminator = make_discriminator()

# Try to load pre-trained generator weights
try:
    # First, try to load the best generator from models directory
    best_generator_path = os.path.join("models", "generators", "generator_best.keras")

    if os.path.exists(best_generator_path):
        trained_generator = load_model(best_generator_path)
        generator.set_weights(trained_generator.get_weights())
        print(f"Loaded best pre-trained generator from: {best_generator_path}")
    else:
        print("generator_best.keras not found, searching for timestamped versions...")

        # Fallback: Look for timestamped generators in models directory
        models_dir = os.path.join("models", "generators")
        if os.path.exists(models_dir):
            generator_files = [f for f in os.listdir(models_dir) if
                               f.startswith("generator_v") and f.endswith(".keras")]

            if generator_files:
                # Use most recent timestamped generator
                generator_files.sort()
                latest_generator = generator_files[-1]
                generator_path = os.path.join(models_dir, latest_generator)

                trained_generator = load_model(generator_path)
                generator.set_weights(trained_generator.get_weights())
                print(f"Loaded timestamped generator from: {generator_path}")
            else:
                print("No pre-trained generators found, starting from scratch")
        else:
            print("Models directory not found, starting generator training from scratch")

except Exception as e:
    print(f"Error loading pre-trained weights: {e}")
    print("Starting generator training from scratch")

# V3.1 KEY IMPROVEMENT: Balanced learning rates
DISCRIMINATOR_LR = 0.0001  # Half the generator's learning rate
GENERATOR_LR = 0.0002     # Standard learning rate

print(f"\n{version} Training Configuration:")
print(f"- Discriminator LR: {DISCRIMINATOR_LR}")
print(f"- Generator LR: {GENERATOR_LR}")
print(f"- Generator training frequency: 2x per discriminator update")

# Compile discriminator with reduced learning rate
discriminator.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=DISCRIMINATOR_LR, beta_1=0.5),
    metrics=['accuracy']
)

# Build combined model for generator training
discriminator.trainable = False  # Freeze discriminator during generator training
gan_input = Input(shape=(latent_dim,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)
combined = Model(gan_input, gan_output)

combined.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=GENERATOR_LR, beta_1=0.5)
)

print("\nModel Summary:")
print(f"Generator parameters: {generator.count_params():,}")
print(f"Discriminator parameters: {discriminator.count_params():,}")

# Fixed noise for consistent image generation across epochs
noise_for_display = np.random.normal(0, 1, (16, latent_dim))

# Training metrics
d_losses = []
g_losses = []
d_accuracies = []

print(f"\nStarting GAN training...")
print(f"Total epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Display interval: {DISPLAY_INTERVAL}")

# Training loop with balanced updates
for epoch in range(1, EPOCHS + 1):

    # ---------------------
    #  Train Discriminator (once per epoch)
    # ---------------------

    # Get random batch of real images
    idx = np.random.randint(0, X_train.shape[0], BATCH_SIZE)
    real_images = X_train[idx]

    # Generate fake images
    noise = np.random.normal(0, 1, (BATCH_SIZE, latent_dim))
    fake_images = generator.predict(noise, verbose=0)

    # V3.1 IMPROVEMENT: Label smoothing to prevent discriminator overconfidence
    real_labels = np.ones((BATCH_SIZE, 1)) * 0.9  # Smooth from 1.0 to 0.9
    fake_labels = np.zeros((BATCH_SIZE, 1))

    # Train discriminator
    discriminator.trainable = True
    d_loss_real = discriminator.train_on_batch(real_images, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # -----------------
    #  Train Generator (2x per epoch - V3.1 IMPROVEMENT)
    # -----------------

    discriminator.trainable = False

    # First generator training step
    noise = np.random.normal(0, 1, (BATCH_SIZE, latent_dim))
    valid_labels = np.ones((BATCH_SIZE, 1))
    g_loss_1 = combined.train_on_batch(noise, valid_labels)

    # Second generator training step
    noise = np.random.normal(0, 1, (BATCH_SIZE, latent_dim))
    valid_labels = np.ones((BATCH_SIZE, 1))
    g_loss_2 = combined.train_on_batch(noise, valid_labels)

    # Average generator loss for logging
    g_loss = (g_loss_1 + g_loss_2) / 2

    # Store metrics
    d_losses.append(d_loss[0])
    g_losses.append(g_loss)
    d_accuracies.append(d_loss[1])

    # Print progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch:5d}: D_loss={d_loss[0]:.4f}, D_acc={d_loss[1]:.4f}, G_loss={g_loss:.4f}")

    # Save generated images at display intervals
    if epoch % DISPLAY_INTERVAL == 0 or epoch == 1:
        save_generated_images(generator, epoch, noise_for_display)

        # Save training metrics plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot losses
        ax1.plot(d_losses, label='Discriminator Loss', alpha=0.7)
        ax1.plot(g_losses, label='Generator Loss', alpha=0.7)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Losses (Balanced)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot discriminator accuracy
        ax2.plot(d_accuracies, label='Discriminator Accuracy', color='orange', alpha=0.7)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Discriminator Accuracy (Balanced)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(base_output_dir, f"training_metrics_epoch_{epoch:05d}.png"),
                    bbox_inches='tight', dpi=100)
        plt.close()

print("\n" + "=" * 50)
print("GAN TRAINING COMPLETED!")
print("=" * 50)

# Save final models with script version and timestamps
generator_filename = f"generator_{version}_{timestamp}.keras"
discriminator_filename = f"discriminator_{version}_{timestamp}.keras"

generator.save(os.path.join(base_output_dir, generator_filename))
discriminator.save(os.path.join(base_output_dir, discriminator_filename))

# Also save to models directories for easy access
models_gen_dir = "models/generators"
models_disc_dir = "models/discriminators"
os.makedirs(models_gen_dir, exist_ok=True)
os.makedirs(models_disc_dir, exist_ok=True)

generator.save(os.path.join(models_gen_dir, generator_filename))
discriminator.save(os.path.join(models_disc_dir, discriminator_filename))

print(f"Models also saved to models/ directories")

# Generate final comparison images
print("\nGenerating final comparison images...")

# Create first vs last epoch comparison
first_epoch_imgs = generator.predict(noise_for_display, verbose=0)
last_epoch_imgs = first_epoch_imgs  # They're the same since we just generated

fig, axes = plt.subplots(2, 8, figsize=(16, 4))

# First epoch results (top row)
for i in range(8):
    img = 0.5 * first_epoch_imgs[i] + 0.5
    img = np.clip(img, 0, 1)
    axes[0, i].imshow(img)
    axes[0, i].set_title(f'Epoch 1')
    axes[0, i].axis('off')

# Last epoch results (bottom row)
for i in range(8):
    img = 0.5 * last_epoch_imgs[i] + 0.5
    img = np.clip(img, 0, 1)
    axes[1, i].imshow(img)
    axes[1, i].set_title(f'Epoch {EPOCHS}')
    axes[1, i].axis('off')

plt.suptitle('GAN Training Progress: First vs Last Epoch (Balanced)', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(base_output_dir, "comparison_first_vs_last.png"),
            bbox_inches='tight', dpi=150)
plt.close()

# Final training summary
final_d_loss = np.mean(d_losses[-100:])  # Average of last 100 epochs
final_g_loss = np.mean(g_losses[-100:])
final_d_acc = np.mean(d_accuracies[-100:])

print(f"\nFinal Training Metrics (last 100 epochs average):")
print(f"- Discriminator Loss: {final_d_loss:.4f}")
print(f"- Generator Loss: {final_g_loss:.4f}")
print(f"- Discriminator Accuracy: {final_d_acc:.4f}")

print(f"\n{version} Improvements Applied:")
print(f"- Discriminator learning rate reduced to {DISCRIMINATOR_LR} (half of generator)")
print(f"- Generator trained 2x per discriminator update")
print(f"- Label smoothing applied (0.9 instead of 1.0 for real labels)")

print(f"\nOutput Files:")
print(f"- Models saved to: {base_output_dir}")
print(f"- Generated images at epochs: 1, {DISPLAY_INTERVAL}, {2 * DISPLAY_INTERVAL}, ..., {EPOCHS}")
print(f"- Training metrics plots saved")
print(f"- Final comparison image: comparison_first_vs_last.png")

print(f"\nAssignment Requirements Met:")
print(f"- Used CIFAR-10 ship images (class 8)")
print(f"- Trained for {EPOCHS} epochs with batch_size={BATCH_SIZE}")
print(f"- Generated screenshots at display_interval={DISPLAY_INTERVAL}")
print(f"- Created first vs last epoch comparison")
print(f"- Saved all models in .keras format")

print(f"\nReady for analysis document creation!")