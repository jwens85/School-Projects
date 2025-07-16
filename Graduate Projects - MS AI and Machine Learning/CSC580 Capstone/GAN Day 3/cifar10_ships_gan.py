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

# Setup
os.makedirs("images", exist_ok=True)
os.makedirs("weights", exist_ok=True)

IMG_SHAPE = (32, 32, 3)
LATENT_DIM = 100

def build_generator():
    """Build improved DCGAN generator for 32x32 CIFAR10 images"""
    
    noise = Input(shape=(LATENT_DIM,))
    
    # Dense layer and reshape to start conv layers
    x = Dense(4 * 4 * 512, use_bias=False)(noise)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Reshape((4, 4, 512))(x)
    
    # Upsample to 8x8
    x = Conv2DTranspose(256, 4, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    # Upsample to 16x16
    x = Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    # Upsample to 32x32
    x = Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    # Final layer to RGB
    img = Conv2DTranspose(3, 4, strides=1, padding='same', activation='tanh')(x)
    
    return Model(noise, img, name="generator")

def build_discriminator():
    """Build improved DCGAN discriminator for 32x32 CIFAR10 images"""
    
    img = Input(shape=IMG_SHAPE)
    
    # Input 32x32x3 -> 16x16x64
    x = Conv2D(64, 4, strides=2, padding='same')(img)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    
    # 16x16x64 -> 8x8x128
    x = Conv2D(128, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    
    # 8x8x128 -> 4x4x256
    x = Conv2D(256, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    
    # 4x4x256 -> 2x2x512
    x = Conv2D(512, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    
    # Flatten and output
    x = Flatten()(x)
    validity = Dense(1, activation='sigmoid')(x)
    
    return Model(img, validity, name="discriminator")

def load_ships_data():
    """Load only ship images from CIFAR10 (class 8)"""
    (X, y), (_, _) = cifar10.load_data()
    X_ships = X[y.flatten() == 8]  # class 8 = ship
    X_ships = X_ships.astype('float32') / 127.5 - 1.0  # Normalize to [-1, 1]
    print(f"Loaded {X_ships.shape[0]} ship images")
    return X_ships

def train_gan(epochs=15000, batch_size=32, sample_interval=1000):
    """Train the DCGAN on CIFAR10 ships with improved training procedure"""
    
    # Load data
    X_ships = load_ships_data()
    
    # Build models
    generator = build_generator()
    discriminator = build_discriminator()
    
    # Compile discriminator with slightly different learning rates
    d_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    discriminator.compile(
        loss='binary_crossentropy',
        optimizer=d_optimizer,
        metrics=['accuracy']
    )
    
    # Build combined model for generator training
    noise = Input(shape=(LATENT_DIM,))
    gen_img = generator(noise)
    
    # For combined model, we freeze discriminator weights
    discriminator.trainable = False
    validity = discriminator(gen_img)
    
    g_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    combined = Model(noise, validity, name="combined")
    combined.compile(loss='binary_crossentropy', optimizer=g_optimizer)
    
    # Training labels with label smoothing
    valid_smooth = np.ones((batch_size, 1)) * 0.9
    fake_smooth = np.zeros((batch_size, 1)) + 0.1
    valid_gen = np.ones((batch_size, 1))  # Generator wants discriminator to output 1
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        # Make discriminator trainable for its own training
        discriminator.trainable = True
        
        # Select random real images
        idx = np.random.randint(0, X_ships.shape[0], batch_size)
        real_imgs = X_ships[idx]
        
        # Generate fake images
        noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
        fake_imgs = generator.predict(noise, verbose=0)
        
        # Train discriminator on real and fake images separately
        d_loss_real = discriminator.train_on_batch(real_imgs, valid_smooth)
        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_smooth)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # -----------------
        #  Train Generator
        # -----------------
        
        # Freeze discriminator for generator training
        discriminator.trainable = False
        
        # Generate new noise for generator training
        noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
        
        # Train generator (wants discriminator to think fake images are real)
        g_loss = combined.train_on_batch(noise, valid_gen)
        
        # Print progress
        if epoch % 100 == 0:
            print(f"{epoch} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")
        
        # Save sample images and model weights
        if epoch % sample_interval == 0:
            save_sample_images(generator, epoch)
            generator.save_weights(f"weights/generator_ships_{epoch}.weights.h5")
            discriminator.save_weights(f"weights/discriminator_ships_{epoch}.weights.h5")

def save_sample_images(generator, epoch, rows=4, cols=4):
    """Generate and save sample images"""
    noise = np.random.normal(0, 1, (rows * cols, LATENT_DIM))
    gen_imgs = generator.predict(noise, verbose=0)
    
    # Rescale images to [0, 1]
    gen_imgs = 0.5 * gen_imgs + 0.5
    
    fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
    for i, img in enumerate(gen_imgs):
        row, col = i // cols, i % cols
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"images/ships_epoch_{epoch}.png")
    plt.close()
    print(f"Saved sample images for epoch {epoch}")

if __name__ == "__main__":
    print("Starting CIFAR10 Ships GAN training...")
    train_gan()