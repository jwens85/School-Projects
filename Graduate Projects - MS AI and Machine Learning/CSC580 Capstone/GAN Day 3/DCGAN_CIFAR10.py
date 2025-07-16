# dcgan_horse_cifar.py

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Input, Dense, Reshape, Flatten,
                                     Dropout, BatchNormalization, LeakyReLU,
                                     Conv2D, Conv2DTranspose)
from tensorflow.keras.optimizers import Adam

# --- Setup ---
os.makedirs("images", exist_ok=True)
os.makedirs("weights", exist_ok=True)

IMG_SHAPE = (32, 32, 3)
LATENT_DIM = 100


# --- Generator (simple + tanh output) ---
def build_generator():
    model = Sequential([
        Dense(256 * 4 * 4, input_dim=LATENT_DIM),
        LeakyReLU(negative_slope=0.2),
        Reshape((4, 4, 256)),
        BatchNormalization(momentum=0.8),
        Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'),
        LeakyReLU(negative_slope=0.2),
        BatchNormalization(momentum=0.8),
        Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'),
        LeakyReLU(negative_slope=0.2),
        BatchNormalization(momentum=0.8),
        Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh'),
    ])
    noise = Input(shape=(LATENT_DIM,))
    img = model(noise)
    return Model(noise, img, name="generator")


# --- Discriminator (simple with dropout) ---
def build_discriminator():
    model = Sequential([
        Conv2D(64, kernel_size=3, strides=2, input_shape=IMG_SHAPE, padding='same'),
        LeakyReLU(negative_slope=0.2),
        Dropout(0.3),
        Conv2D(128, kernel_size=3, strides=2, padding='same'),
        LeakyReLU(negative_slope=0.2),
        Dropout(0.3),
        Flatten(),
        Dense(1, activation='sigmoid'),
    ])
    img = Input(shape=IMG_SHAPE)
    validity = model(img)
    return Model(img, validity, name="discriminator")


# --- Load only HORSE images ---
(X, y), (_, _) = cifar10.load_data()
X = X[y.flatten() == 7]  # class 7 = horse
X = X.astype('float32') / 127.5 - 1.0

# --- Build models ---
generator = build_generator()
discriminator = build_discriminator()

# --- Compile discriminator only ---
opt_d = Adam(learning_rate=0.0002, beta_1=0.5, clipvalue=1.0)
discriminator.compile(loss='binary_crossentropy', optimizer=opt_d, metrics=['accuracy'])

# --- Build combined model ---
noise = Input(shape=(LATENT_DIM,))
gen_img = generator(noise)
discriminator.trainable = False
validity = discriminator(gen_img)
combined = Model(noise, validity, name="dcgan")
opt_g = Adam(learning_rate=0.0001, beta_1=0.5, clipvalue=1.0)
combined.compile(loss='binary_crossentropy', optimizer=opt_g)


# --- Training ---
def train(epochs=30000, batch_size=64, d_steps=5, sample_interval=1000):
    valid = np.ones((batch_size, 1)) * 0.9
    fake = np.zeros((batch_size, 1)) + np.random.rand(batch_size, 1) * 0.1

    for epoch in range(epochs):
        # Train D multiple times
        for _ in range(d_steps):
            idx = np.random.randint(0, X.shape[0], batch_size)
            real_imgs = X[idx]
            noise_vec = np.random.normal(0, 1, (batch_size, LATENT_DIM))
            gen_imgs = generator.predict(noise_vec, verbose=0)

            d_loss_real = discriminator.train_on_batch(real_imgs, valid)
            d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)

        # Train Generator once
        noise_vec = np.random.normal(0, 1, (batch_size, LATENT_DIM))
        g_loss = combined.train_on_batch(noise_vec, valid)

        # Output
        if epoch % sample_interval == 0:
            print(f"{epoch} [D loss real: {d_loss_real[0]:.4f}, fake: {d_loss_fake[0]:.4f}] [G loss: {g_loss:.4f}]")
            save_images(epoch)
            generator.save_weights(f"weights/generator_{epoch}.weights.h5")
            discriminator.save_weights(f"weights/discriminator_{epoch}.weights.h5")


def save_images(epoch, rows=5, cols=5):
    noise = np.random.normal(0, 1, (rows * cols, LATENT_DIM))
    imgs = generator.predict(noise, verbose=0)
    imgs = 0.5 * imgs + 0.5
    fig, axes = plt.subplots(rows, cols, figsize=(5, 5))
    for i, img in enumerate(imgs):
        axes[i // cols, i % cols].imshow(img)
        axes[i // cols, i % cols].axis('off')
    plt.tight_layout()
    plt.savefig(f"images/horse_{epoch}.png")
    plt.close()


if __name__ == "__main__":
    train()
