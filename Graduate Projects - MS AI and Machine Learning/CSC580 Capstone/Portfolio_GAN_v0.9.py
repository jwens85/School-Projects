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

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Parameters
image_shape = (32, 32, 3)
latent_dim = 100

# Output directory
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
base_output_dir = os.path.join("outputs", "v0.9", timestamp)
os.makedirs(base_output_dir, exist_ok=True)

# Load CIFAR-10 ships
(X, y), (_, _) = cifar10.load_data()
X = X[y.flatten() == 8]  # Ships only
print(f"Loaded {len(X)} ship images")

# Normalize to [-1, 1] for tanh
X_train = (X.astype(np.float32) - 127.5) / 127.5
print(f"Data range: [{X_train.min():.2f}, {X_train.max():.2f}]")


def make_generator():
    """Simple but effective generator"""
    model = Sequential()

    # Project and reshape
    model.add(Dense(8 * 8 * 256, input_dim=latent_dim, use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Reshape((8, 8, 256)))

    # Upsample to 16x16
    model.add(Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    # Upsample to 32x32
    model.add(Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model


def make_discriminator():
    """Simple but effective discriminator"""
    model = Sequential()

    # Downsample to 16x16
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=image_shape))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    # Downsample to 8x8
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    # Output
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model


# Build models
print("Building GAN...")
generator = make_generator()
discriminator = make_discriminator()

print("\nGenerator:")
generator.summary()
print("\nDiscriminator:")
discriminator.summary()

# CRITICAL: Use different learning rates - this is often the key!
d_optimizer = Adam(learning_rate=0.0001, beta_1=0.5)  # Slower for discriminator
g_optimizer = Adam(learning_rate=0.0004, beta_1=0.5)  # Faster for generator

# Compile discriminator
discriminator.compile(optimizer=d_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Build GAN (discriminator frozen)
discriminator.trainable = False
gan_input = Input(shape=(latent_dim,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)
gan = Model(gan_input, gan_output)
gan.compile(optimizer=g_optimizer, loss='binary_crossentropy')

# Training parameters
EPOCHS = 1000  # Start small to see if it works
BATCH_SIZE = 64  # Try larger batch
SAVE_INTERVAL = 100

print(f"\nStarting training: {EPOCHS} epochs, batch size {BATCH_SIZE}")

# Fixed noise for consistent evaluation
fixed_noise = np.random.normal(0, 1, (16, latent_dim))


def save_images(epoch, noise, filename):
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
        plt.suptitle('Generated Ship Images - First Epoch (Before Training)')
    else:
        plt.suptitle(f'Generated Ship Images - Epoch {epoch}')

    plt.tight_layout()
    filepath = os.path.join(base_output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=100)
    plt.close()
    return filepath


# Save some real images for reference
real_imgs = X_train[:16]
real_imgs_display = 0.5 * real_imgs + 0.5
fig, axs = plt.subplots(4, 4, figsize=(8, 8))
for i in range(16):
    row, col = i // 4, i % 4
    axs[row, col].imshow(real_imgs_display[i])
    axs[row, col].axis('off')
plt.suptitle('Real CIFAR-10 Ship Images (Target)')
plt.tight_layout()
plt.savefig(os.path.join(base_output_dir, "real_ships_target.png"), bbox_inches='tight', dpi=100)
plt.close()

# Training tracking
d_losses, g_losses = [], []
d_real_acc, d_fake_acc = [], []

# Save first epoch (before training)
print("Saving first epoch images...")
save_images(0, fixed_noise, "epoch_0_first_epoch.png")

print("Starting training loop...")

for epoch in range(EPOCHS):

    # ===========================
    # Train Discriminator
    # ===========================

    # Get real images
    idx = np.random.randint(0, X_train.shape[0], BATCH_SIZE)
    real_images = X_train[idx]

    # Generate fake images
    noise = np.random.normal(0, 1, (BATCH_SIZE, latent_dim))
    fake_images = generator.predict(noise, verbose=0)

    # Create labels (with label smoothing for stability)
    real_labels = np.ones((BATCH_SIZE, 1)) * 0.9  # Smooth real labels
    fake_labels = np.zeros((BATCH_SIZE, 1)) + 0.1  # Smooth fake labels

    # Train discriminator on real images
    d_loss_real = discriminator.train_on_batch(real_images, real_labels)

    # Train discriminator on fake images
    d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)

    # Average the losses
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # ===========================
    # Train Generator
    # ===========================

    # Generate noise
    noise = np.random.normal(0, 1, (BATCH_SIZE, latent_dim))

    # Train generator (wants discriminator to classify fakes as real)
    valid_labels = np.ones((BATCH_SIZE, 1))  # Generator wants these to be "real"
    g_loss = gan.train_on_batch(noise, valid_labels)

    # Store metrics
    d_losses.append(d_loss[0])
    g_losses.append(g_loss)
    d_real_acc.append(d_loss_real[1])
    d_fake_acc.append(d_loss_fake[1])

    # Print progress
    if epoch % 50 == 0:
        print(f"Epoch {epoch:4d} | D_loss: {d_loss[0]:.4f} | D_acc: {d_loss[1] * 100:5.1f}% | G_loss: {g_loss:.4f}")
        print(f"          | D_real: {d_loss_real[1] * 100:5.1f}% | D_fake: {d_loss_fake[1] * 100:5.1f}%")

        # Check for problems
        if d_loss_real[1] > 0.95:
            print("WARNING: Discriminator too strong on real images!")
        if d_loss_fake[1] < 0.05:
            print("WARNING: Discriminator too weak on fake images!")
        if g_loss > 5.0:
            print("WARNING: Generator loss very high!")

    # Save sample images
    if epoch % SAVE_INTERVAL == 0 and epoch > 0:
        save_images(epoch, fixed_noise, f"epoch_{epoch}_progress.png")

# Save final epoch
print("\nTraining completed!")
save_images(EPOCHS, fixed_noise, f"epoch_{EPOCHS}_final_epoch.png")

# Create analysis plots
print("Creating analysis plots...")
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

# Loss curves
ax1.plot(d_losses, label='Discriminator', alpha=0.8, linewidth=1)
ax1.plot(g_losses, label='Generator', alpha=0.8, linewidth=1)
ax1.set_title('Training Losses')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Discriminator accuracy
ax2.plot(d_real_acc, label='Real Images', alpha=0.8, linewidth=1)
ax2.plot(d_fake_acc, label='Fake Images', alpha=0.8, linewidth=1)
ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
ax2.set_title('Discriminator Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Loss balance
loss_diff = np.array(d_losses) - np.array(g_losses)
ax3.plot(loss_diff, alpha=0.8, linewidth=1)
ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax3.set_title('Loss Balance (D - G)')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss Difference')
ax3.grid(True, alpha=0.3)

# Generator gradient check (approximate)
g_gradient_proxy = np.diff(g_losses)
ax4.plot(g_gradient_proxy, alpha=0.8, linewidth=1)
ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax4.set_title('Generator Learning Rate (Loss Diff)')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Loss Change')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(base_output_dir, "training_diagnostics.png"), bbox_inches='tight', dpi=100)
plt.close()

# Final diagnosis
print("\n" + "=" * 60)
print("TRAINING DIAGNOSIS")
print("=" * 60)

final_d_real = d_real_acc[-1]
final_d_fake = d_fake_acc[-1]
final_g_loss = g_losses[-1]
final_d_loss = d_losses[-1]

print(f"Final discriminator accuracy on real: {final_d_real * 100:.1f}%")
print(f"Final discriminator accuracy on fake: {final_d_fake * 100:.1f}%")
print(f"Final generator loss: {final_g_loss:.3f}")
print(f"Final discriminator loss: {final_d_loss:.3f}")

print("\nDIAGNOSIS:")
if final_d_real > 0.95:
    print("PROBLEM: Discriminator is too strong - generator can't fool it")
    print("SOLUTION: Use slower discriminator learning rate or train generator more")
elif final_d_real < 0.6:
    print("PROBLEM: Discriminator is too weak - not providing good feedback")
    print("SOLUTION: Use faster discriminator learning rate")
else:
    print("GOOD: Discriminator strength seems reasonable")

if final_g_loss > 3.0:
    print("PROBLEM: Generator loss is very high - it's struggling to learn")
    print("SOLUTION: Check architecture or use different loss function")
elif np.std(g_losses[-100:]) < 0.01:
    print("PROBLEM: Generator loss has flatlined - it stopped learning")
    print("SOLUTION: Need better architecture or training strategy")
else:
    print("GOOD: Generator seems to be learning")

# Check for actual learning
g_loss_trend = np.mean(g_losses[-100:]) - np.mean(g_losses[:100])
if g_loss_trend < -0.5:
    print("GOOD: Generator loss decreased over training - it learned something!")
elif abs(g_loss_trend) < 0.1:
    print("WARNING: Generator loss didn't change much - minimal learning")
else:
    print("PROBLEM: Generator loss increased - training instability")

print(f"\nFiles saved to: {base_output_dir}")
print("Check the images to see if there's any actual learning happening!")
print("=" * 60)