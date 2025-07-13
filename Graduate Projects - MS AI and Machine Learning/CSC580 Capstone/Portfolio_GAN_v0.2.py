import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import (
    Input, Dense, Reshape, Flatten, Dropout,
    BatchNormalization, Activation, LeakyReLU,
    Conv2D, UpSampling2D, Conv2DTranspose
)
from keras.optimizers import Adam

# Define image dimensions and training parameters
image_shape = (32, 32, 3)
latent_dimensions = 100
epochs = 5000
batch_size = 32  # Reduced for better training stability
save_interval = 500

# Timestamped output directory (e.g., outputs/v0.2/2025-07-13_0932/)
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
base_output_dir = os.path.join("outputs", "v0.2", timestamp)
os.makedirs(base_output_dir, exist_ok=True)

# Load CIFAR-10 and extract only airplane images (class label 0)
def load_airplane_data():
    (X, y), (_, _) = cifar10.load_data()
    X = X[y.flatten() == 0]
    X = (X.astype(np.float32) - 127.5) / 127.5  # Normalize to [-1, 1]
    return X

# Define the improved generator model
def build_generator():
    model = Sequential()
    
    # Start with 4x4 base instead of 8x8 for smoother upsampling
    model.add(Dense(256 * 4 * 4, activation="relu", input_dim=latent_dimensions))
    model.add(Reshape((4, 4, 256)))
    model.add(BatchNormalization(momentum=0.8))
    
    # First upsampling: 4x4 -> 8x8
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    
    # Second upsampling: 8x8 -> 16x16
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    
    # Third upsampling: 16x16 -> 32x32
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    
    # Final convolution for RGB output
    model.add(Conv2D(3, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    noise = Input(shape=(latent_dimensions,))
    image = model(noise)
    return Model(noise, image)

# Define the improved discriminator model
def build_discriminator():
    model = Sequential()
    
    # First conv layer
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=image_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    
    # Second conv layer
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))
    
    # Third conv layer
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))
    
    # Fourth conv layer
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
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
    gen_images = generator.predict(noise, verbose=0)

    for i, img in enumerate(gen_images):
        img = 0.5 * img + 0.5  # Rescale [-1, 1] to [0, 1]
        img = np.clip(img, 0, 1)  # Ensure values are in valid range
        filename = f"epoch_{epoch}_sample_{i}.png"
        filepath = os.path.join(output_dir, filename)
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
        plt.close()

# Train the GAN with improved stability
def train_gan():
    X_train = load_airplane_data()
    print(f"Training on {len(X_train)} airplane images")

    # Build discriminator with slower learning rate
    discriminator = build_discriminator()
    discriminator.compile(loss="binary_crossentropy", 
                         optimizer=Adam(learning_rate=0.0001, beta_1=0.5), 
                         metrics=["accuracy"])

    # Build generator with faster learning rate
    generator = build_generator()

    z = Input(shape=(latent_dimensions,))
    img = generator(z)
    discriminator.trainable = False
    valid = discriminator(img)

    combined = Model(z, valid)
    combined.compile(loss="binary_crossentropy", 
                    optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

    # Training loop with improved stability
    for epoch in range(1, epochs + 1):
        # Train discriminator
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dimensions))
        gen_imgs = generator.predict(noise, verbose=0)

        # Use softer labels for better training
        real_labels = np.ones((batch_size, 1)) * 0.9  # Label smoothing
        fake_labels = np.zeros((batch_size, 1)) + 0.1  # Noisy labels

        d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train generator twice for every discriminator update
        for _ in range(2):
            noise = np.random.normal(0, 1, (batch_size, latent_dimensions))
            valid_y = np.ones((batch_size, 1))
            g_loss = combined.train_on_batch(noise, valid_y)

        # Print progress
        if epoch % 100 == 0:
            print(f"[Epoch {epoch:>5}]  D_loss: {d_loss[0]:.4f}  D_acc: {d_loss[1]*100:5.2f}%  G_loss: {g_loss:.4f}")

        # Save sample images
        if epoch % save_interval == 0:
            save_images(generator, epoch)

    # Save final models
    generator.save(os.path.join(base_output_dir, "generator_final.h5"))
    discriminator.save(os.path.join(base_output_dir, "discriminator_final.h5"))
    print(f"Models saved to {base_output_dir}")

# Run the script
if __name__ == '__main__':
    train_gan()