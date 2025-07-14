import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import (
    Input, Dense, Reshape, Flatten, Dropout,
    BatchNormalization, LeakyReLU,
    Conv2D, Conv2DTranspose
)
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

image_shape = (32, 32, 3)
latent_dim = 64
EPOCHS = 15000
BATCH_SIZE = 64
DISPLAY_INTERVAL = 2500

script_name = os.path.basename(__file__)
script_name_clean = script_name.replace('.py', '')
name_parts = script_name_clean.split('_')
if len(name_parts) >= 4:
    version = f"{name_parts[2]}_{name_parts[3].lower()}"
else:
    version = name_parts[2]

timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
base_output_dir = os.path.join("outputs", version, timestamp)
os.makedirs(base_output_dir, exist_ok=True)

(X, y), (_, _) = cifar10.load_data()
X = X[y.flatten() == 8]
X_train = X.astype(np.float32) / 255.0

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i in range(8):
    row = i // 4
    col = i % 4
    img = X_train[i]
    axes[row, col].imshow(img)
    axes[row, col].set_title(f'Real Ship {i + 1}')
    axes[row, col].axis('off')
plt.suptitle('Real CIFAR-10 Ships After Preprocessing', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(base_output_dir, "debug_real_ships.png"), bbox_inches='tight', dpi=150)
plt.close()

def make_generator():
    model = Sequential()
    model.add(Dense(8 * 8 * 256, input_dim=latent_dim))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Reshape((8, 8, 256)))
    model.add(Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', activation='sigmoid'))
    return model

def make_discriminator():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=4, strides=2, input_shape=image_shape, padding="same"))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=4, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=4, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    return model

def save_generated_images(generator, epoch, noise_for_display):
    generated_images = generator.predict(noise_for_display, verbose=0)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(16):
        row = i // 4
        col = i % 4
        img = np.clip(generated_images[i], 0, 1)
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
    plt.suptitle(f'Generated Ships - Epoch {epoch}', fontsize=16)
    plt.tight_layout()
    filename = f"generated_epoch_{epoch:05d}.png"
    filepath = os.path.join(base_output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close()

generator = make_generator()
discriminator = make_discriminator()

d_schedule = ExponentialDecay(0.00005, decay_steps=2500, decay_rate=0.5)
g_schedule = ExponentialDecay(0.00005, decay_steps=2500, decay_rate=0.5)

d_optimizer = Adam(learning_rate=d_schedule, beta_1=0.5, clipnorm=1.0)
g_optimizer = Adam(learning_rate=g_schedule, beta_1=0.5, clipnorm=1.0)

discriminator.compile(loss='binary_crossentropy', optimizer=d_optimizer, metrics=['accuracy'])
discriminator.trainable = False
gan_input = Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
combined = Model(gan_input, gan_output)
combined.compile(loss='binary_crossentropy', optimizer=g_optimizer)

noise_for_display = np.random.normal(0, 1, (16, latent_dim))

d_losses = []
g_losses = []
d_real_accuracies = []
d_fake_accuracies = []

for epoch in range(1, EPOCHS + 1):
    idx = np.random.randint(0, X_train.shape[0], BATCH_SIZE)
    real_images = X_train[idx]

    noise = np.random.normal(0, 1, (BATCH_SIZE, latent_dim))
    fake_images = generator.predict(noise, verbose=0)

    real_labels = np.ones((BATCH_SIZE, 1)) * 0.9
    fake_labels = np.zeros((BATCH_SIZE, 1))
    if np.random.rand() < 0.05:
        real_labels, fake_labels = fake_labels, real_labels

    discriminator.trainable = True
    d_loss_real = discriminator.train_on_batch(real_images, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    discriminator.trainable = False
    noise = np.random.normal(0, 1, (BATCH_SIZE, latent_dim))
    valid_labels = np.ones((BATCH_SIZE, 1))
    g_loss_1 = combined.train_on_batch(noise, valid_labels)
    noise = np.random.normal(0, 1, (BATCH_SIZE, latent_dim))
    g_loss_2 = combined.train_on_batch(noise, valid_labels)
    g_loss = (g_loss_1 + g_loss_2) / 2

    d_losses.append(d_loss[0])
    g_losses.append(g_loss)
    d_real_accuracies.append(d_loss_real[1])
    d_fake_accuracies.append(d_loss_fake[1])

    if epoch % 100 == 0:
        print(f"Epoch {epoch:5d}: D_loss={d_loss[0]:.4f}, Real_acc={d_loss_real[1]:.3f}, Fake_acc={d_loss_fake[1]:.3f}, G_loss={g_loss:.4f}")

    if epoch % DISPLAY_INTERVAL == 0 or epoch == 1:
        save_generated_images(generator, epoch, noise_for_display)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(d_losses, label='Discriminator Loss', alpha=0.7)
        ax1.plot(g_losses, label='Generator Loss', alpha=0.7)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax2.plot(d_real_accuracies, label='Real Accuracy', color='green')
        ax2.plot(d_fake_accuracies, label='Fake Accuracy', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(base_output_dir, f"training_metrics_epoch_{epoch:05d}.png"), bbox_inches='tight', dpi=100)
        plt.close()
