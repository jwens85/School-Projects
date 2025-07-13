import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)


class CIFAR10GAN:
    def __init__(self, latent_dim=100, img_shape=(32, 32, 3)):
        self.latent_dim = latent_dim
        self.img_shape = img_shape

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            metrics=['accuracy']
        )

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = keras.Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model (stacked generator and discriminator)
        self.combined = keras.Model(z, validity)
        self.combined.compile(
            loss='binary_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        )

    def build_generator(self):
        model = keras.Sequential([
            # Foundation for 4x4 feature maps
            layers.Dense(4 * 4 * 512, use_bias=False, input_shape=(self.latent_dim,)),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Reshape((4, 4, 512)),

            # Upsample to 8x8
            layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),

            # Upsample to 16x16
            layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),

            # Upsample to 32x32
            layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),

            # Output layer
            layers.Conv2DTranspose(3, (4, 4), strides=(1, 1), padding='same',
                                   use_bias=False, activation='tanh')
        ])

        return model

    def build_discriminator(self):
        model = keras.Sequential([
            # Input layer
            layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same',
                          input_shape=self.img_shape),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),

            # Downsample to 8x8
            layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),

            # Downsample to 4x4
            layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),

            # Downsample to 2x2
            layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),

            # Flatten and output
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid')
        ])

        return model

    def load_data(self):
        # Load CIFAR-10 dataset
        (x_train, _), (_, _) = keras.datasets.cifar10.load_data()

        # Normalize to [-1, 1] range (important for tanh activation)
        x_train = (x_train.astype(np.float32) - 127.5) / 127.5

        return x_train

    def train(self, epochs=10000, batch_size=128, save_interval=1000):
        # Load the dataset
        x_train = self.load_data()

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # Create directory for saving images
        os.makedirs('gan_images', exist_ok=True)

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]

            # Generate noise
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of fake images
            gen_imgs = self.generator.predict(noise, verbose=0)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Generate noise
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Print progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, D Loss: {d_loss[0]:.4f}, D Acc: {d_loss[1]:.4f}, G Loss: {g_loss:.4f}")

            # Save generated images at intervals
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise, verbose=0)

        # Rescale images to [0, 1]
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c, figsize=(10, 10))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].axis('off')
                cnt += 1
        fig.suptitle(f"Generated Images - Epoch {epoch}")
        plt.tight_layout()
        plt.savefig(f"gan_images/cifar10_gan_epoch_{epoch}.png")
        plt.close()

    def generate_images(self, num_images=25):
        """Generate and display random images"""
        noise = np.random.normal(0, 1, (num_images, self.latent_dim))
        gen_imgs = self.generator.predict(noise, verbose=0)

        # Rescale images to [0, 1]
        gen_imgs = 0.5 * gen_imgs + 0.5

        # Display images
        fig, axes = plt.subplots(5, 5, figsize=(10, 10))
        for i in range(5):
            for j in range(5):
                axes[i, j].imshow(gen_imgs[i * 5 + j])
                axes[i, j].axis('off')
        plt.tight_layout()
        plt.show()


# Usage example
if __name__ == "__main__":
    # Initialize the GAN
    gan = CIFAR10GAN()

    # Print model summaries
    print("Generator Summary:")
    gan.generator.summary()
    print("\nDiscriminator Summary:")
    gan.discriminator.summary()

    # Train the GAN
    # Start with fewer epochs for testing, increase for better results
    gan.train(epochs=5000, batch_size=128, save_interval=500)

    # Generate some images after training
    gan.generate_images()

    # Save the trained models
    gan.generator.save('cifar10_generator.h5')
    gan.discriminator.save('cifar10_discriminator.h5')
    print("Models saved successfully!")