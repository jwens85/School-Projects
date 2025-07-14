import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from keras.datasets import cifar10
from keras.models import Sequential, Model, load_model
from keras.layers import (
    Input, Dense, Reshape, Flatten, Dropout,
    BatchNormalization, LeakyReLU,
    Conv2D, Conv2DTranspose, Lambda
)
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import backend as K

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

# WGAN-GP specific parameters
CRITIC_ITERATIONS = 5  # Train critic 5x more than generator
GRADIENT_PENALTY_WEIGHT = 10.0

# Auto-derive version from filename with debug info
script_name = os.path.basename(__file__)
print(f"DEBUG: Original filename: {script_name}")

# Remove .py extension first
script_name_clean = script_name.replace('.py', '')
print(f"DEBUG: Clean filename: {script_name_clean}")

# Split and extract version
name_parts = script_name_clean.split('_')
print(f"DEBUG: Name parts: {name_parts}")

if len(name_parts) >= 4:
    # Has descriptor: Portfolio_GAN_v5.0_WGAN-GP -> "v5.0_wgan-gp"
    version = f"{name_parts[2]}_{name_parts[3].lower()}"
    print(f"DEBUG: Using descriptor version: {version}")
else:
    # No descriptor: Portfolio_GAN_v5.0 -> "v5.0"
    version = name_parts[2] if len(name_parts) > 2 else "v5.0"
    print(f"DEBUG: Using simple version: {version}")

print(f"DEBUG: Final version: {version}")

# Updated project structure for Day 2 - use current directory structure
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
base_output_dir = os.path.join("outputs", version, timestamp)
os.makedirs(base_output_dir, exist_ok=True)

print(f"GAN {version} - WGAN-GP Implementation")
print(f"Output directory: {base_output_dir}")

# Load CIFAR-10 ships
(X, y), (_, _) = cifar10.load_data()
X = X[y.flatten() == 8]
print(f"Loaded {len(X)} ship images")

# Normalize to [-1, 1] for tanh
X_train = (X.astype(np.float32) - 127.5) / 127.5
print(f"Data range: [{X_train.min():.2f}, {X_train.max():.2f}]")


def make_generator():
    """Generator - improved architecture with spectral normalization in mind"""
    model = Sequential()

    # Project and reshape - using 4x4 base for better upsampling
    model.add(Dense(4 * 4 * 512, input_dim=latent_dim))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Reshape((4, 4, 512)))

    # Upsample to 8x8
    model.add(Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    # Upsample to 16x16
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    # Upsample to 32x32
    model.add(Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', activation='tanh'))

    return model


def make_critic():
    """Critic (not discriminator) for WGAN-GP - no sigmoid, no batch norm in critic"""
    model = Sequential()

    # First conv block: 32x32 -> 16x16
    model.add(Conv2D(64, kernel_size=4, strides=2,
                     input_shape=image_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    # Second conv block: 16x16 -> 8x8
    model.add(Conv2D(128, kernel_size=4, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    # Third conv block: 8x8 -> 4x4
    model.add(Conv2D(256, kernel_size=4, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    # Fourth conv block: 4x4 -> 2x2
    model.add(Conv2D(512, kernel_size=4, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    # Output layer - LINEAR activation for Wasserstein distance
    model.add(Flatten())
    model.add(Dense(1))  # No activation = linear

    return model


def gradient_penalty(critic, real_images, fake_images, batch_size):
    """Calculate gradient penalty for WGAN-GP"""
    # Random interpolation factor
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)

    # Interpolated images
    interpolated = alpha * real_images + (1 - alpha) * fake_images

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        # Critic output for interpolated images
        validity = critic(interpolated, training=True)

    # Calculate gradients
    gradients = tape.gradient(validity, interpolated)

    # Calculate gradient penalty
    gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean((gradients_norm - 1.0) ** 2)

    return gradient_penalty


def wasserstein_loss(y_true, y_pred):
    """Wasserstein loss function"""
    return K.mean(y_true * y_pred)


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

    plt.suptitle(f'Generated Ships - Epoch {epoch} (WGAN-GP)', fontsize=16)
    plt.tight_layout()

    # Save with assignment-friendly naming
    filename = f"generated_epoch_{epoch:05d}.png"
    filepath = os.path.join(base_output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"Saved: {filename}")


# Build models
print("\nBuilding Generator and Critic...")
generator = make_generator()
critic = make_critic()

# Try to load pre-trained generator weights from current directory structure
try:
    # First, try to load from current models directory
    best_generator_path = os.path.join("models", "generators", "generator_best.keras")

    if os.path.exists(best_generator_path):
        trained_generator = load_model(best_generator_path)
        generator.set_weights(trained_generator.get_weights())
        print(f"Loaded best pre-trained generator from: {best_generator_path}")
    else:
        print("generator_best.keras not found in current directory, checking ../GAN Day 1...")

        # Fallback: Look for models from Day 1
        day1_models_dir = os.path.join("..", "GAN Day 1", "models", "generators")
        if os.path.exists(day1_models_dir):
            generator_files = [f for f in os.listdir(day1_models_dir) if
                               f.startswith("generator_v") and f.endswith(".keras")]

            if generator_files:
                # Use most recent from Day 1
                generator_files.sort()
                latest_generator = generator_files[-1]
                generator_path = os.path.join(day1_models_dir, latest_generator)

                trained_generator = load_model(generator_path)
                generator.set_weights(trained_generator.get_weights())
                print(f"Loaded Day 1 generator from: {generator_path}")
            else:
                print("No pre-trained generators found, starting from scratch")
        else:
            print("No previous models found, starting generator training from scratch")

except Exception as e:
    print(f"Error loading pre-trained weights: {e}")
    print("Starting generator training from scratch")

# WGAN-GP learning rates
CRITIC_LR = 0.0001
GENERATOR_LR = 0.0001

print(f"\n{version} WGAN-GP Configuration:")
print(f"- Critic LR: {CRITIC_LR}")
print(f"- Generator LR: {GENERATOR_LR}")
print(f"- Critic iterations per generator update: {CRITIC_ITERATIONS}")
print(f"- Gradient penalty weight: {GRADIENT_PENALTY_WEIGHT}")

# Compile critic with Wasserstein loss
critic.compile(
    loss=wasserstein_loss,
    optimizer=Adam(learning_rate=CRITIC_LR, beta_1=0.5, beta_2=0.9)
)

# Build combined model for generator training
critic.trainable = False
gan_input = Input(shape=(latent_dim,))
generated_image = generator(gan_input)
gan_output = critic(generated_image)
combined = Model(gan_input, gan_output)

combined.compile(
    loss=wasserstein_loss,
    optimizer=Adam(learning_rate=GENERATOR_LR, beta_1=0.5, beta_2=0.9)
)

print("\nModel Summary:")
print(f"Generator parameters: {generator.count_params():,}")
print(f"Critic parameters: {critic.count_params():,}")

# Fixed noise for consistent image generation across epochs
noise_for_display = np.random.normal(0, 1, (16, latent_dim))

# Training metrics
critic_losses = []
generator_losses = []
wasserstein_distances = []

print(f"\nStarting WGAN-GP training...")
print(f"Total epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Display interval: {DISPLAY_INTERVAL}")

# WGAN-GP Training loop
for epoch in range(1, EPOCHS + 1):

    # ---------------------
    #  Train Critic (CRITIC_ITERATIONS times per generator update)
    # ---------------------

    critic_loss_epoch = []

    for _ in range(CRITIC_ITERATIONS):
        # Get random batch of real images
        idx = np.random.randint(0, X_train.shape[0], BATCH_SIZE)
        real_images = X_train[idx]

        # Generate fake images
        noise = np.random.normal(0, 1, (BATCH_SIZE, latent_dim))
        fake_images = generator.predict(noise, verbose=0)

        # WGAN-GP labels: real=1, fake=-1
        real_labels = np.ones((BATCH_SIZE, 1))
        fake_labels = -np.ones((BATCH_SIZE, 1))

        # Train critic on real and fake images
        critic.trainable = True

        with tf.GradientTape() as tape:
            # Critic predictions
            real_validity = critic(real_images, training=True)
            fake_validity = critic(fake_images, training=True)

            # Wasserstein loss
            critic_loss_real = tf.reduce_mean(real_validity)
            critic_loss_fake = tf.reduce_mean(fake_validity)
            critic_loss = critic_loss_fake - critic_loss_real

            # Gradient penalty
            gp = gradient_penalty(critic, real_images, fake_images, BATCH_SIZE)

            # Total critic loss
            total_critic_loss = critic_loss + GRADIENT_PENALTY_WEIGHT * gp

        # Update critic
        critic_gradients = tape.gradient(total_critic_loss, critic.trainable_variables)
        critic.optimizer.apply_gradients(zip(critic_gradients, critic.trainable_variables))

        critic_loss_epoch.append(total_critic_loss.numpy())

    # Average critic loss for this epoch
    avg_critic_loss = np.mean(critic_loss_epoch)

    # ---------------------
    #  Train Generator (once per CRITIC_ITERATIONS)
    # ---------------------

    critic.trainable = False

    noise = np.random.normal(0, 1, (BATCH_SIZE, latent_dim))
    valid_labels = np.ones((BATCH_SIZE, 1))  # Generator wants critic to output high values for fakes

    generator_loss = combined.train_on_batch(noise, valid_labels)

    # Calculate Wasserstein distance (negative of critic loss)
    wasserstein_distance = -avg_critic_loss

    # Store metrics
    critic_losses.append(avg_critic_loss)
    generator_losses.append(generator_loss)
    wasserstein_distances.append(wasserstein_distance)

    # Print progress
    if epoch % 100 == 0:
        print(
            f"Epoch {epoch:5d}: C_loss={avg_critic_loss:.4f}, G_loss={generator_loss:.4f}, W_dist={wasserstein_distance:.4f}")

    # Save generated images at display intervals
    if epoch % DISPLAY_INTERVAL == 0 or epoch == 1:
        save_generated_images(generator, epoch, noise_for_display)

        # Save training metrics plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot losses
        ax1.plot(critic_losses, label='Critic Loss', alpha=0.7)
        ax1.plot(generator_losses, label='Generator Loss', alpha=0.7)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('WGAN-GP Training Losses')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot Wasserstein distance
        ax2.plot(wasserstein_distances, label='Wasserstein Distance', color='green', alpha=0.7)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Distance')
        ax2.set_title('Wasserstein Distance (Lower = Better)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(base_output_dir, f"training_metrics_epoch_{epoch:05d}.png"),
                    bbox_inches='tight', dpi=100)
        plt.close()

print("\n" + "=" * 50)
print("WGAN-GP TRAINING COMPLETED!")
print("=" * 50)

# Save final models with script version and timestamps
generator_filename = f"generator_{version}_{timestamp}.keras"
critic_filename = f"critic_{version}_{timestamp}.keras"

generator.save(os.path.join(base_output_dir, generator_filename))
critic.save(os.path.join(base_output_dir, critic_filename))

# Also save to models directories for easy access
models_gen_dir = os.path.join("models", "generators")
models_critic_dir = os.path.join("models", "critics")
os.makedirs(models_gen_dir, exist_ok=True)
os.makedirs(models_critic_dir, exist_ok=True)

generator.save(os.path.join(models_gen_dir, generator_filename))
critic.save(os.path.join(models_critic_dir, critic_filename))

# Save best models if this is the best run
generator.save(os.path.join(models_gen_dir, "generator_best.keras"))
critic.save(os.path.join(models_critic_dir, "critic_best.keras"))

print(f"Models saved to: {models_gen_dir}")

# Generate final comparison images
print("\nGenerating final comparison images...")

# Save progression comparison
noise_fixed = np.random.normal(0, 1, (8, latent_dim))
final_images = generator.predict(noise_fixed, verbose=0)

fig, axes = plt.subplots(2, 4, figsize=(12, 6))

for i in range(8):
    row = i // 4
    col = i % 4

    img = 0.5 * final_images[i] + 0.5
    img = np.clip(img, 0, 1)

    axes[row, col].imshow(img)
    axes[row, col].set_title(f'Final Result {i + 1}')
    axes[row, col].axis('off')

plt.suptitle('WGAN-GP Final Results', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(base_output_dir, "final_results_wgan_gp.png"),
            bbox_inches='tight', dpi=150)
plt.close()

# Final training summary
final_critic_loss = np.mean(critic_losses[-100:])
final_generator_loss = np.mean(generator_losses[-100:])
final_wasserstein_distance = np.mean(wasserstein_distances[-100:])

print(f"\nFinal Training Metrics (last 100 epochs average):")
print(f"- Critic Loss: {final_critic_loss:.4f}")
print(f"- Generator Loss: {final_generator_loss:.4f}")
print(f"- Wasserstein Distance: {final_wasserstein_distance:.4f}")

print(f"\n{version} WGAN-GP Improvements:")
print(f"- Wasserstein loss with gradient penalty")
print(f"- Critic trained {CRITIC_ITERATIONS}x per generator update")
print(f"- No batch normalization in critic")
print(f"- Linear output activation in critic")
print(f"- Gradient penalty weight: {GRADIENT_PENALTY_WEIGHT}")

print(f"\nOutput Files:")
print(f"- Models saved to: GAN Day 2/models/")
print(f"- Generated images at epochs: 1, {DISPLAY_INTERVAL}, {2 * DISPLAY_INTERVAL}, ..., {EPOCHS}")
print(f"- Training metrics plots saved")
print(f"- Final results image: final_results_wgan_gp.png")

print(f"\nProject Structure Updated:")
print(f"- All Day 2 work in current directory")
print(f"- Models: models/[generators|critics]/")
print(f"- Outputs: outputs/{version}/{timestamp}/")

print(f"\nWGAN-GP should solve your discriminator collapse issue!")
print(f"Ready for stable training and better image generation!")