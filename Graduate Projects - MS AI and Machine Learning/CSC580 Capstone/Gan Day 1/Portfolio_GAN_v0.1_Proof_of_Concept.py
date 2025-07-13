import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import (
    Input, Dense, Reshape, Flatten, Dropout,
    BatchNormalization, Activation, LeakyReLU,
    Conv2D, UpSampling2D
)
from keras.optimizers import Adam

# Define image dimensions and training parameters
image_shape = (32, 32, 3)
latent_dimensions = 100
epochs = 5000
batch_size = 64
save_interval = 500

# Timestamped output directory (e.g., outputs/v0.1/2025-07-13_0932/)
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
base_output_dir = os.path.join("outputs", "v0.1", timestamp)
os.makedirs(base_output_dir, exist_ok=True)

# Load CIFAR-10 and extract only airplane images (class label 0)
def load_airplane_data():
    (X, y), (_, _) = cifar10.load_data()
    X = X[y.flatten() == 0]
    X = (X.astype(np.float32) - 127.5) / 127.5  # Normalize to [-1, 1]
    return X

# Define the generator model
def build_generator():
    model = Sequential()
    model.add(Dense(128 * 8 * 8, activation="relu", input_dim=latent_dimensions))
    model.add(Reshape((8, 8, 128)))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.78))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.78))
    model.add(Activation("relu"))
    model.add(Conv2D(3, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    noise = Input(shape=(latent_dimensions,))
    image = model(noise)
    return Model(noise, image)

# Define the discriminator model
def build_discriminator():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=image_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    image = Input(shape=image_shape)
    validity = model(image)
    return Model(image, validity)

# Save generated images to the timestamped output folder
def save_images(generator, epoch, output_dir=base_output_dir):
    noise = np.random.normal(0, 1, (5, latent_dimensions))
    gen_images = generator.predict(noise)

    for i, img in enumerate(gen_images):
        img = 0.5 * img + 0.5  # Rescale [-1, 1] to [0, 1]
        filename = f"epoch_{epoch}_sample_{i}.png"
        filepath = os.path.join(output_dir, filename)
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(filepath)
        plt.close()

# Train the GAN
def train_gan():
    X_train = load_airplane_data()

    discriminator = build_discriminator()
    discriminator.compile(loss="binary_crossentropy", optimizer=Adam(0.0002, 0.5), metrics=["accuracy"])

    generator = build_generator()

    z = Input(shape=(latent_dimensions,))
    img = generator(z)
    discriminator.trainable = False
    valid = discriminator(img)

    combined = Model(z, valid)
    combined.compile(loss="binary_crossentropy", optimizer=Adam(0.0002, 0.5))

    for epoch in range(1, epochs + 1):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dimensions))
        gen_imgs = generator.predict(noise)

        real_labels = np.ones((batch_size, 1)) * 0.9  # Label smoothing
        fake_labels = np.zeros((batch_size, 1))

        d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, latent_dimensions))
        valid_y = np.ones((batch_size, 1))
        g_loss = combined.train_on_batch(noise, valid_y)

        if epoch % 100 == 0:
            print(f"[Epoch {epoch:>5}]  D_loss: {d_loss[0]:.4f}  D_acc: {d_loss[1]*100:5.2f}%  G_loss: {g_loss:.4f}")

        if epoch % save_interval == 0:
            save_images(generator, epoch)

# Run the script
if __name__ == '__main__':
    train_gan()
