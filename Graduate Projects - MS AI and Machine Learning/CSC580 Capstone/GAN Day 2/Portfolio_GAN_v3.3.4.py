
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization
from keras.optimizers import Adam
import tensorflow as tf
from datetime import datetime

# Set up output directory
base_output_dir = "./GAN_Outputs"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(base_output_dir, f"gan_run_{timestamp}")
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)

# Load CIFAR-10 and filter for class 8 (ships)
(X_train, y_train), (_, _) = cifar10.load_data()
X_train = X_train[y_train.flatten() == 8]
X_train = X_train.astype("float32") / 255.0

# Define input shape and latent dimension
img_shape = (32, 32, 3)
latent_dim = 100

# Build the generator
def build_generator():
    model = Sequential()
    model.add(Dense(8 * 8 * 256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((8, 8, 256)))
    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"))
    return model

# Build the discriminator
def build_discriminator():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same", input_shape=img_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))
    return model

# Create and compile models
optimizer = Adam(0.0002, 0.5)
discriminator = build_discriminator()
discriminator.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
generator = build_generator()

# Combined model
z = tf.keras.Input(shape=(latent_dim,))
img = generator(z)
discriminator.trainable = False
valid = discriminator(img)
combined = tf.keras.Model(z, valid)
combined.compile(loss="binary_crossentropy", optimizer=optimizer)

# Training loop
def train(epochs, batch_size=64, save_interval=500):
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    fixed_noise = np.random.normal(0, 1, (16, latent_dim))

    for epoch in range(1, epochs + 1):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = generator.predict(noise, verbose=0)

        d_loss_real = discriminator.train_on_batch(real_imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = combined.train_on_batch(noise, valid)

        if epoch % save_interval == 0:
            print(f"{epoch} [D loss: {d_loss[0]:.4f}, acc: {100*d_loss[1]:.2f}] [G loss: {g_loss:.4f}]")
            save_images(epoch, generator, fixed_noise)
            generator.save(os.path.join(output_dir, "models", f"generator_epoch_{epoch:05d}.h5"))
            discriminator.save(os.path.join(output_dir, "models", f"discriminator_epoch_{epoch:05d}.h5"))

# Save generated images for visualization
def save_images(epoch, generator, fixed_noise):
    gen_imgs = generator.predict(fixed_noise, verbose=0)
    fig, axs = plt.subplots(4, 4, figsize=(6,6))
    for i in range(16):
        axs[i // 4, i % 4].imshow(gen_imgs[i])
        axs[i // 4, i % 4].axis("off")
    plt.suptitle(f"Generated Ships - Epoch {epoch}")
    fig.savefig(os.path.join(output_dir, "images", f"generated_epoch_{epoch:05d}.png"))
    plt.close()

# Run training
train(epochs=10000, batch_size=64, save_interval=500)
