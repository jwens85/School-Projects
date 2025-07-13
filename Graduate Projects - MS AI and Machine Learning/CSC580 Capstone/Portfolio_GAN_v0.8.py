import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import (
    Input, Dense, Reshape, Flatten, Dropout,
    BatchNormalization, Activation, LeakyReLU,
    Conv2D, Conv2DTranspose
)
from keras.optimizers import Adam
import tensorflow as tf

# Set seeds for reproducibility
np.random.seed(123)
tf.random.set_seed(123)

# Parameters
image_shape = (32, 32, 3)
latent_dim = 100

# Output directory
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
base_output_dir = os.path.join("outputs", "v0.8", timestamp)
os.makedirs(base_output_dir, exist_ok=True)

# Load data
(X, y), (_, _) = cifar10.load_data()
X = X[y.flatten() == 8]  # Ships
print(f"Loaded {len(X)} ship images")

# Show real images
fig, axs = plt.subplots(4, 4, figsize=(8, 8))
for i in range(16):
    row, col = i // 4, i % 4
    axs[row, col].imshow(X[i])
    axs[row, col].axis('off')
plt.suptitle('Real CIFAR-10 Ship Images')
plt.tight_layout()
plt.savefig(os.path.join(base_output_dir, "real_ships.png"), bbox_inches='tight', dpi=100)
plt.close()


def make_generator():
    """Ultra-simple generator"""
    model = Sequential([
        # Start: 100 -> 8*8*256
        Dense(8 * 8 * 256, input_dim=latent_dim),
        LeakyReLU(0.2),
        Reshape((8, 8, 256)),

        # 8x8 -> 16x16
        Conv2DTranspose(128, 4, strides=2, padding='same'),
        BatchNormalization(),
        LeakyReLU(0.2),

        # 16x16 -> 32x32
        Conv2DTranspose(3, 4, strides=2, padding='same'),
        Activation('tanh')
    ])
    return model


def make_discriminator():
    """Ultra-simple discriminator"""
    model = Sequential([
        # 32x32 -> 16x16
        Conv2D(64, 4, strides=2, padding='same', input_shape=image_shape),
        LeakyReLU(0.2),
        Dropout(0.3),

        # 16x16 -> 8x8
        Conv2D(128, 4, strides=2, padding='same'),
        BatchNormalization(),
        LeakyReLU(0.2),
        Dropout(0.3),

        # 8x8 -> 1
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    return model


# Build models
print("Building simple GAN...")
generator = make_generator()
discriminator = make_discriminator()

print("\nGenerator:")
generator.summary()
print("\nDiscriminator:")
discriminator.summary()

# Compile discriminator
discriminator.compile(
    optimizer=Adam(0.0002, 0.5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Build GAN
discriminator.trainable = False
gan_input = Input(shape=(latent_dim,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)
gan = Model(gan_input, gan_output)

gan.compile(
    optimizer=Adam(0.0002, 0.5),
    loss='binary_crossentropy'
)

# Training setup
EPOCHS = 2000  # Even fewer epochs
BATCH_SIZE = 32  # Smaller batch
SAMPLE_INTERVAL = 200

# Normalize data
X_train = (X.astype(np.float32) - 127.5) / 127.5
print(f"Data shape: {X_train.shape}, Range: [{X_train.min():.2f}, {X_train.max():.2f}]")

# Fixed noise for tracking progress
fixed_noise = np.random.normal(0, 1, (16, latent_dim))


def save_generated_images(epoch, noise, path):
    """Save generated images"""
    gen_imgs = generator.predict(noise, verbose=0)
    gen_imgs = 0.5 * gen_imgs + 0.5  # Scale to [0,1]
    gen_imgs = np.clip(gen_imgs, 0, 1)

    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(16):
        row, col = i // 4, i % 4
        axs[row, col].imshow(gen_imgs[i])
        axs[row, col].axis('off')

    if epoch == 0:
        plt.suptitle('Generated Images - First Epoch (Before Training)')
    else:
        plt.suptitle(f'Generated Images - Epoch {epoch}')

    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', dpi=100)
    plt.close()


# Training metrics
d_losses, g_losses = [], []
d_real_acc, d_fake_acc = [], []

print(f"\nStarting training for {EPOCHS} epochs...")

# Save first epoch
save_generated_images(0, fixed_noise, os.path.join(base_output_dir, "epoch_0_first_epoch.png"))

# Training loop
for epoch in range(EPOCHS):

    # Train Discriminator
    # Real images
    idx = np.random.randint(0, X_train.shape[0], BATCH_SIZE)
    real_imgs = X_train[idx]

    # Fake images
    noise = np.random.normal(0, 1, (BATCH_SIZE, latent_dim))
    fake_imgs = generator.predict(noise, verbose=0)

    # Labels (with label smoothing)
    real_labels = np.ones((BATCH_SIZE, 1)) * 0.9
    fake_labels = np.zeros((BATCH_SIZE, 1)) + 0.1

    # Train discriminator
    d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train Generator
    noise = np.random.normal(0, 1, (BATCH_SIZE, latent_dim))
    valid_labels = np.ones((BATCH_SIZE, 1))  # Generator wants discriminator to think these are real

    g_loss = gan.train_on_batch(noise, valid_labels)

    # Save metrics
    d_losses.append(d_loss[0])
    g_losses.append(g_loss)
    d_real_acc.append(d_loss_real[1])
    d_fake_acc.append(d_loss_fake[1])

    # Print progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch:>4} | D_loss: {d_loss[0]:.3f} | D_acc: {d_loss[1] * 100:5.1f}% | G_loss: {g_loss:.3f}")
        print(f"          | D_real_acc: {d_loss_real[1] * 100:5.1f}% | D_fake_acc: {d_loss_fake[1] * 100:5.1f}%")

    # Save sample images
    if epoch % SAMPLE_INTERVAL == 0 and epoch > 0:
        save_generated_images(epoch, fixed_noise,
                              os.path.join(base_output_dir, f"epoch_{epoch}_progress.png"))

# Save final results
print("\nTraining completed!")
save_generated_images(EPOCHS, fixed_noise,
                      os.path.join(base_output_dir, f"epoch_{EPOCHS}_last_epoch.png"))

# Create training plots
print("Creating training analysis...")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Losses
axes[0, 0].plot(d_losses, label='Discriminator', alpha=0.7)
axes[0, 0].plot(g_losses, label='Generator', alpha=0.7)
axes[0, 0].set_title('Training Losses')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Discriminator accuracies
axes[0, 1].plot(d_real_acc, label='Real Images', alpha=0.7)
axes[0, 1].plot(d_fake_acc, label='Fake Images', alpha=0.7)
axes[0, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
axes[0, 1].set_title('Discriminator Accuracy')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Loss difference
loss_diff = np.array(d_losses) - np.array(g_losses)
axes[1, 0].plot(loss_diff, alpha=0.7)
axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Balance')
axes[1, 0].set_title('Loss Balance (D - G)')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss Difference')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Moving averages
window = 50
if len(d_losses) > window:
    d_smooth = np.convolve(d_losses, np.ones(window) / window, mode='valid')
    g_smooth = np.convolve(g_losses, np.ones(window) / window, mode='valid')
    axes[1, 1].plot(d_smooth, label='D (smooth)', alpha=0.7)
    axes[1, 1].plot(g_smooth, label='G (smooth)', alpha=0.7)
    axes[1, 1].set_title('Smoothed Losses')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(base_output_dir, "training_analysis.png"), bbox_inches='tight', dpi=100)
plt.close()

# Generate some variety samples
variety_noise = np.random.normal(0, 1, (16, latent_dim))
save_generated_images(EPOCHS, variety_noise,
                      os.path.join(base_output_dir, "final_variety.png"))

# Save models
generator.save(os.path.join(base_output_dir, "simple_generator.h5"))
discriminator.save(os.path.join(base_output_dir, "simple_discriminator.h5"))

# Final analysis
print("\n" + "=" * 60)
print("SIMPLE GAN TRAINING RESULTS")
print("=" * 60)
print(f"Architecture: Ultra-minimal GAN")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Final D loss: {d_losses[-1]:.3f}")
print(f"Final G loss: {g_losses[-1]:.3f}")
print(f"Final D accuracy: {(d_real_acc[-1] + d_fake_acc[-1]) / 2 * 100:.1f}%")

# Quality assessment
recent_d_real = np.mean(d_real_acc[-50:])
recent_d_fake = np.mean(d_fake_acc[-50:])
recent_balance = np.mean(np.abs(loss_diff[-50:]))

print(f"\nQuality Indicators (last 50 epochs):")
print(f"D accuracy on real: {recent_d_real * 100:.1f}%")
print(f"D accuracy on fake: {recent_d_fake * 100:.1f}%")
print(f"Average loss balance: {recent_balance:.3f}")

if 0.6 <= recent_d_real <= 0.9 and 0.1 <= recent_d_fake <= 0.4:
    print("STATUS: Discriminator performance looks reasonable")
elif recent_d_real > 0.95:
    print("WARNING: Discriminator may be too strong")
elif recent_d_real < 0.5:
    print("WARNING: Discriminator may be too weak")

if recent_balance < 0.5:
    print("STATUS: Generator and discriminator appear balanced")
else:
    print("WARNING: Training may be unstable")

print(f"\nFiles created:")
print(f"- real_ships.png (reference)")
print(f"- epoch_0_first_epoch.png (ASSIGNMENT)")
print(f"- epoch_{EPOCHS}_last_epoch.png (ASSIGNMENT)")
print(f"- training_analysis.png")
print(f"- final_variety.png")
print("=" * 60)