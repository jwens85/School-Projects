import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
import datetime

# Set parameters
latent_dim = 100
img_shape = (32, 32, 3)
epochs = 2500
batch_size = 64
n_critic = 3  # Discriminator updates per generator update

# Create output directory
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H%M')
output_dir = f"outputs/v3.3.1/{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Load and preprocess CIFAR-10 (class 8 = ship)
(x_train, y_train), (_, _) = cifar10.load_data()
x_train = x_train[y_train.flatten() == 8]
x_train = (x_train.astype(np.float32) - 127.5) / 127.5  # Normalize to [-1, 1]

# Build Generator
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(8*8*256, use_bias=False, input_shape=(latent_dim,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((8, 8, 256)),
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(3, kernel_size=3, strides=1, padding='same', activation='tanh')
    ])
    return model

# Build Discriminator
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, kernel_size=3, strides=2, padding='same', input_shape=img_shape),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(128, kernel_size=3, strides=2, padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Instantiate models
generator = build_generator()
discriminator = build_discriminator()

# Optimizers
gen_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
disc_optimizer = Adam(learning_rate=0.00002, beta_1=0.5)

# Loss
cross_entropy = tf.keras.losses.BinaryCrossentropy()

# Trackers
g_losses, d_losses = [], []
real_accs, fake_accs = [], []

# Training loop
for epoch in range(1, epochs + 1):
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    real_imgs = x_train[idx]

    # Add Gaussian noise to real images
    real_imgs += tf.random.normal(tf.shape(real_imgs), mean=0.0, stddev=0.1)

    # Real and fake labels (label smoothing for real)
    real_labels = tf.ones((batch_size, 1)) * 0.9
    fake_labels = tf.zeros((batch_size, 1))

    # Generate fake images
    noise = tf.random.normal((batch_size, latent_dim))
    fake_imgs = generator(noise, training=True)

    # Train Discriminator
    for _ in range(n_critic):
        with tf.GradientTape() as tape:
            real_pred = discriminator(real_imgs, training=True)
            fake_pred = discriminator(fake_imgs, training=True)
            d_loss_real = cross_entropy(real_labels, real_pred)
            d_loss_fake = cross_entropy(fake_labels, fake_pred)
            d_loss = d_loss_real + d_loss_fake
        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        disc_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

    # Train Generator
    noise = tf.random.normal((batch_size, latent_dim))
    with tf.GradientTape() as tape:
        gen_imgs = generator(noise, training=True)
        fake_pred = discriminator(gen_imgs, training=True)
        g_loss = cross_entropy(real_labels, fake_pred)
    grads = tape.gradient(g_loss, generator.trainable_variables)
    gen_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

    # Logging
    g_losses.append(g_loss.numpy())
    d_losses.append(d_loss.numpy())
    real_accs.append(np.mean(real_pred.numpy() > 0.5))
    fake_accs.append(np.mean(fake_pred.numpy() < 0.5))

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: D_loss={d_loss.numpy():.4f}, Real_acc={real_accs[-1]:.3f}, "
              f"Fake_acc={fake_accs[-1]:.3f}, G_loss={g_loss.numpy():.4f}")

        # Save generated image
        sample_noise = tf.random.normal((1, latent_dim))
        generated_img = generator(sample_noise, training=False)[0]
        img = ((generated_img + 1) * 127.5).numpy().astype(np.uint8)
        plt.imsave(f"{output_dir}/generated_epoch_{epoch:05}.png", img)

# Plot loss and accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(d_losses, label='Discriminator Loss')
plt.plot(g_losses, label='Generator Loss')
plt.title("Training Losses (Dynamic LR)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(real_accs, label='Real Image Accuracy', color='green')
plt.plot(fake_accs, label='Fake Image Accuracy', color='red')
plt.title("Discriminator Accuracies (Separate)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig(f"{output_dir}/training_metrics_epoch_{epochs:05}.png")
plt.close()
