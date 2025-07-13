import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os


# Simple, working CIFAR-10 GAN
class SimpleCIFAR10GAN:
    def __init__(self):
        self.latent_dim = 100
        self.img_shape = (32, 32, 3)

        # Build generator
        self.generator = self.make_generator()

        # Build discriminator
        self.discriminator = self.make_discriminator()

        # Compile discriminator
        self.discriminator.compile(
            optimizer=keras.optimizers.Adam(0.0002, 0.5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Combined model for training generator
        z = keras.Input(shape=(self.latent_dim,))
        img = self.generator(z)
        self.discriminator.trainable = False
        valid = self.discriminator(img)

        self.combined = keras.Model(z, valid)
        self.combined.compile(
            optimizer=keras.optimizers.Adam(0.0002, 0.5),
            loss='binary_crossentropy'
        )

    def make_generator(self):
        model = keras.Sequential([
            layers.Dense(8 * 8 * 256, input_dim=self.latent_dim),
            layers.Reshape((8, 8, 256)),

            layers.UpSampling2D(),
            layers.Conv2D(128, 3, padding='same'),
            layers.BatchNormalization(momentum=0.8),
            layers.Activation('relu'),

            layers.UpSampling2D(),
            layers.Conv2D(64, 3, padding='same'),
            layers.BatchNormalization(momentum=0.8),
            layers.Activation('relu'),

            layers.Conv2D(3, 3, padding='same'),
            layers.Activation('tanh')
        ])
        return model

    def make_discriminator(self):
        model = keras.Sequential([
            layers.Conv2D(32, 3, strides=2, input_shape=self.img_shape, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.25),

            layers.Conv2D(64, 3, strides=2, padding='same'),
            layers.ZeroPadding2D(padding=((0, 1), (0, 1))),
            layers.BatchNormalization(momentum=0.8),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.25),

            layers.Conv2D(128, 3, strides=2, padding='same'),
            layers.BatchNormalization(momentum=0.8),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.25),

            layers.Conv2D(256, 3, strides=1, padding='same'),
            layers.BatchNormalization(momentum=0.8),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.25),

            layers.Flatten(),
            layers.Dense(1, activation='sigmoid')
        ])
        return model

    def train(self, epochs=30000, batch_size=32, save_interval=2000):
        # Load CIFAR-10
        (X_train, _), (_, _) = keras.datasets.cifar10.load_data()
        X_train = X_train / 127.5 - 1.0  # Normalize to [-1, 1]

        # Create output directory
        os.makedirs('images', exist_ok=True)

        # Ground truth labels
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # Train Discriminator
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise, verbose=0)

            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train Generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, valid)

            # Print progress
            if epoch % 500 == 0:
                print(f"Epoch {epoch}, D Loss: {d_loss[0]:.4f}, D Acc: {d_loss[1]:.4f}, G Loss: {g_loss:.4f}")

            # Save images
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise, verbose=0)

        # Rescale to [0, 1]
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c, figsize=(5, 5))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].axis('off')
                cnt += 1
        fig.suptitle(f"Epoch {epoch}")
        plt.savefig(f"images/cifar10_{epoch}.png")
        plt.close()


# Just run it
if __name__ == "__main__":
    gan = SimpleCIFAR10GAN()

    print("Generator:")
    gan.generator.summary()
    print("\nDiscriminator:")
    gan.discriminator.summary()

    print("\nStarting training...")
    gan.train(epochs=20000, batch_size=32)

    print("Done!")