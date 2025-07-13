import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import (
    Input, Dense, Reshape, Flatten, Dropout,
    BatchNormalization, Activation, LeakyReLU,
    Conv2D, Conv2DTranspose, ZeroPadding2D
)
from keras.optimizers import Adam
from keras.initializers import RandomNormal
import tensorflow as tf

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define parameters
image_shape = (32, 32, 3)
latent_dimensions = 100

# Timestamped output directory
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
base_output_dir = os.path.join("outputs", "v0.7", timestamp)
os.makedirs(base_output_dir, exist_ok=True)

# Load CIFAR-10 data and select ship class (class 8)
(X, y), (_, _) = cifar10.load_data()
X = X[y.flatten() == 8]
print(f"Loaded {len(X)} ship images")

# Show real ship images for reference
print("Saving sample real ship images for reference...")
fig, axs = plt.subplots(4, 4, figsize=(8, 8))
for i in range(16):
    row, col = i // 4, i % 4
    axs[row, col].imshow(X[i])
    axs[row, col].axis('off')
plt.suptitle('Real CIFAR-10 Ship Images (What GAN Should Learn)')
plt.tight_layout()
real_ships_path = os.path.join(base_output_dir, "real_ship_reference.png")
plt.savefig(real_ships_path, bbox_inches='tight', dpi=100)
plt.close()


def build_generator():
    """
    Build a working generator using Conv2DTranspose (much better than UpSampling2D + Conv2D)
    """
    model = Sequential()

    # Start with dense layer
    model.add(Dense(4 * 4 * 512, input_dim=latent_dimensions, use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Reshape((4, 4, 512)))

    # 4x4 -> 8x8
    model.add(Conv2DTranspose(256, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    # 8x8 -> 16x16
    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    # 16x16 -> 32x32
    model.add(Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(Activation('tanh'))

    return model


def build_discriminator():
    """
    Build a working discriminator
    """
    model = Sequential()

    # 32x32 -> 16x16
    model.add(Conv2D(64, kernel_size=4, strides=2, padding='same', input_shape=image_shape))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    # 16x16 -> 8x8
    model.add(Conv2D(128, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    # 8x8 -> 4x4
    model.add(Conv2D(256, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    # 4x4 -> 1x1
    model.add(Conv2D(1, kernel_size=4, strides=1, padding='valid'))
    model.add(Flatten())
    model.add(Activation('sigmoid'))

    return model


# Build models
print("Building working GAN architecture...")
generator = build_generator()
discriminator = build_discriminator()

print("Generator Summary:")
generator.summary()
print("\nDiscriminator Summary:")
discriminator.summary()

# Compile discriminator
discriminator.compile(
    optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Build combined model
discriminator.trainable = False
z = Input(shape=(latent_dimensions,))
img = generator(z)
validity = discriminator(img)
combined = Model(z, validity)

combined.compile(
    optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
    loss='binary_crossentropy'
)

# Training parameters
epochs = 3000  # Reasonable number
batch_size = 64  # Larger batch size
save_interval = 500

# Normalize data to [-1, 1] for tanh activation
X_train = (X.astype(np.float32) - 127.5) / 127.5
print(f"Data normalized to range: [{X_train.min():.2f}, {X_train.max():.2f}]")

# Fixed noise for consistent evaluation
np.random.seed(42)
fixed_noise = np.random.normal(0, 1, (16, latent_dimensions))


def save_images(epoch, generator, noise, filename):
    """Save generated images"""
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale to [0, 1]

    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    count = 0
    for i in range(4):
        for j in range(4):
            axs[i, j].imshow(gen_imgs[count])
            axs[i, j].axis('off')
            count += 1

    if epoch == 0:
        plt.suptitle('Generated Ship Images - First Epoch (Before Training)')
    else:
        plt.suptitle(f'Generated Ship Images - Epoch {epoch}')

    plt.tight_layout()
    filepath = os.path.join(base_output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=100)
    plt.close()
    return filepath


# Training tracking
d_losses = []
g_losses = []
d_real_acc = []
d_fake_acc = []

print(f"Starting training for {epochs} epochs...")
print(f"Output directory: {base_output_dir}")

# Save first epoch (before training)
first_path = save_images(0, generator, fixed_noise, "epoch_0_first_epoch.png")
print(f"First epoch images saved: {first_path}")

# Training loop
for epoch in range(epochs):

    # ---------------------
    #  Train Discriminator
    # ---------------------

    # Select a random batch of real images
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_imgs = X_train[idx]

    # Generate fake images
    noise = np.random.normal(0, 1, (batch_size, latent_dimensions))
    fake_imgs = generator.predict(noise)

    # Train discriminator on real images
    real_labels = np.ones((batch_size, 1)) * 0.9  # Label smoothing
    d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)

    # Train discriminator on fake images
    fake_labels = np.zeros((batch_size, 1)) + 0.1  # Label smoothing
    d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)

    # Average discriminator loss
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # ---------------------
    #  Train Generator
    # ---------------------

    # Train generator (wants discriminator to mistake fakes as real)
    noise = np.random.normal(0, 1, (batch_size, latent_dimensions))
    valid_y = np.ones((batch_size, 1))  # Generator wants these labeled as real

    g_loss = combined.train_on_batch(noise, valid_y)

    # Save training metrics
    d_losses.append(d_loss[0])
    g_losses.append(g_loss)
    d_real_acc.append(d_loss_real[1])
    d_fake_acc.append(d_loss_fake[1])

    # Print progress
    if epoch % 100 == 0:
        print(
            f"[Epoch {epoch:>4}/{epochs}] D_loss: {d_loss[0]:.4f} D_acc: {d_loss[1] * 100:5.2f}% G_loss: {g_loss:.4f}")

    # Save sample images
    if epoch % save_interval == 0 and epoch > 0:
        save_images(epoch, generator, fixed_noise, f"epoch_{epoch}_progress.png")

# Save final epoch
print("\nTraining completed!")
final_path = save_images(epochs, generator, fixed_noise, f"epoch_{epochs}_last_epoch.png")
print(f"Final epoch images saved: {final_path}")

# Generate additional random samples to show variety
print("Generating additional samples to show variety...")
random_noise = np.random.normal(0, 1, (16, latent_dimensions))
variety_path = save_images(epochs, generator, random_noise, "final_variety_samples.png")

# Training analysis plots
print("Generating training analysis plots...")
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Loss curves
ax1.plot(d_losses, label='Discriminator Loss', alpha=0.7)
ax1.plot(g_losses, label='Generator Loss', alpha=0.7)
ax1.set_title('Training Losses')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Discriminator accuracy
ax2.plot(d_real_acc, label='Real Image Accuracy', alpha=0.7)
ax2.plot(d_fake_acc, label='Fake Image Accuracy', alpha=0.7)
ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random Chance')
ax2.set_title('Discriminator Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Loss difference (balance indicator)
loss_diff = np.array(d_losses) - np.array(g_losses)
ax3.plot(loss_diff, alpha=0.7)
ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Perfect Balance')
ax3.set_title('Loss Difference (D - G)')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss Difference')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Moving averages for smoother view
window = 50
if len(d_losses) > window:
    d_smooth = np.convolve(d_losses, np.ones(window) / window, mode='valid')
    g_smooth = np.convolve(g_losses, np.ones(window) / window, mode='valid')
    ax4.plot(d_smooth, label='D Loss (smoothed)', alpha=0.7)
    ax4.plot(g_smooth, label='G Loss (smoothed)', alpha=0.7)
    ax4.set_title('Smoothed Losses')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

plt.tight_layout()
analysis_path = os.path.join(base_output_dir, "complete_training_analysis.png")
plt.savefig(analysis_path, bbox_inches='tight', dpi=100)
plt.close()

# Save models
generator.save(os.path.join(base_output_dir, "working_generator.h5"))
discriminator.save(os.path.join(base_output_dir, "working_discriminator.h5"))

# Final analysis
print("\n" + "=" * 70)
print("WORKING GAN TRAINING RESULTS")
print("=" * 70)
print(f"Dataset: CIFAR-10 Ship Images (Class 8)")
print(f"Training samples: {len(X_train)}")
print(f"Epochs completed: {epochs}")
print(f"Batch size: {batch_size}")
print("")
print("FINAL METRICS:")
print(f"Final discriminator loss: {d_losses[-1]:.4f}")
print(f"Final generator loss: {g_losses[-1]:.4f}")
print(f"Final D accuracy on real images: {d_real_acc[-1] * 100:.2f}%")
print(f"Final D accuracy on fake images: {d_fake_acc[-1] * 100:.2f}%")
print("")
print("TRAINING QUALITY INDICATORS:")
avg_d_real_acc = np.mean(d_real_acc[-100:])  # Last 100 epochs
avg_d_fake_acc = np.mean(d_fake_acc[-100:])  # Last 100 epochs

if 0.7 <= avg_d_real_acc <= 0.95:
    print("GOOD: Discriminator correctly identifies real images")
else:
    print("WARNING: Discriminator performance on real images may be off")

if 0.05 <= avg_d_fake_acc <= 0.5:
    print("GOOD: Discriminator appropriately challenges generator")
else:
    print("WARNING: Discriminator may be too weak/strong on fake images")

balance_score = abs(np.mean(loss_diff[-100:]))
if balance_score < 0.5:
    print("GOOD: Generator and discriminator are well balanced")
else:
    print("INFO: Some imbalance between generator and discriminator")

print("")
print("KEY ARCHITECTURE IMPROVEMENTS:")
print("- Used Conv2DTranspose instead of UpSampling2D + Conv2D")
print("- Proper kernel sizes (4x4) and strides (2x2)")
print("- Better layer progression: 4->8->16->32")
print("- Appropriate batch normalization placement")
print("- Label smoothing for training stability")
print("- Fixed random seeds for reproducibility")
print("")
print("FILES GENERATED:")
print("- real_ship_reference.png (Target images)")
print("- epoch_0_first_epoch.png (ASSIGNMENT: First epoch)")
print(f"- epoch_{epochs}_last_epoch.png (ASSIGNMENT: Last epoch)")
print("- final_variety_samples.png (Additional samples)")
print("- complete_training_analysis.png (Training metrics)")
print("=" * 70)