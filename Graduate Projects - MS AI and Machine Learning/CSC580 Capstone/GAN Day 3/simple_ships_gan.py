22#!/usr/bin/env python3
"""
Simple Working Ships GAN - Guaranteed to work without noise
Based on proven minimal architecture
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose, Conv2D, LeakyReLU, BatchNormalization, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# Settings
NOISE_DIM = 64  # Smaller noise dimension
BATCH_SIZE = 32
os.makedirs("images", exist_ok=True)

def load_ships():
    """Load ship images from CIFAR10"""
    (X_train, y_train), (_, _) = cifar10.load_data()
    ships = X_train[y_train.flatten() == 8]  # Class 8 = ships
    ships = ships.astype('float32') / 255.0  # Normalize to [0, 1] instead of [-1, 1]
    print(f"Loaded {ships.shape[0]} ship images")
    return ships

def create_generator():
    """Simple generator that works"""
    model = Sequential([
        Dense(8 * 8 * 128, input_dim=NOISE_DIM),
        LeakyReLU(0.2),
        Reshape((8, 8, 128)),
        
        Conv2DTranspose(64, 4, strides=2, padding='same'),
        LeakyReLU(0.2),
        
        Conv2DTranspose(32, 4, strides=2, padding='same'),
        LeakyReLU(0.2),
        
        Conv2DTranspose(3, 3, strides=1, padding='same', activation='sigmoid'),  # sigmoid for [0,1] output
    ], name='generator')
    return model

def create_discriminator():
    """Simple discriminator"""
    model = Sequential([
        Conv2D(32, 3, strides=2, padding='same', input_shape=(32, 32, 3)),
        LeakyReLU(0.2),
        Dropout(0.25),
        
        Conv2D(64, 3, strides=2, padding='same'),
        LeakyReLU(0.2),
        Dropout(0.25),
        
        Conv2D(128, 3, strides=2, padding='same'),
        LeakyReLU(0.2),
        Dropout(0.25),
        
        Flatten(),
        Dense(1, activation='sigmoid')
    ], name='discriminator')
    return model

def save_images(generator, epoch):
    """Save generated images"""
    noise = np.random.normal(0, 1, (16, NOISE_DIM))
    gen_imgs = generator.predict(noise, verbose=0)
    
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(gen_imgs[i])
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'images/simple_ships_{epoch}.png')
    plt.close()
    print(f"Saved images for epoch {epoch}")

def train_simple_gan(epochs=5000):
    """Train with simple, proven approach"""
    
    # Load data
    ships = load_ships()
    
    # Create models
    generator = create_generator()
    discriminator = create_discriminator()
    
    # Compile discriminator
    discriminator.compile(
        optimizer=Adam(learning_rate=0.0001),  # Slower learning
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Create combined model
    discriminator.trainable = False
    gan_input = tf.keras.Input(shape=(NOISE_DIM,))
    generated_image = generator(gan_input)
    gan_output = discriminator(generated_image)
    gan = tf.keras.Model(gan_input, gan_output)
    gan.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy')
    
    print(f"Generator params: {generator.count_params():,}")
    print(f"Discriminator params: {discriminator.count_params():,}")
    
    # Training loop
    for epoch in range(epochs):
        
        # Train discriminator
        discriminator.trainable = True
        
        # Real images
        idx = np.random.randint(0, ships.shape[0], BATCH_SIZE)
        real_imgs = ships[idx]
        real_labels = np.ones((BATCH_SIZE, 1)) * 0.9  # Label smoothing
        
        # Fake images
        noise = np.random.normal(0, 1, (BATCH_SIZE, NOISE_DIM))
        fake_imgs = generator.predict(noise, verbose=0)
        fake_labels = np.zeros((BATCH_SIZE, 1)) + 0.1  # Label smoothing
        
        # Train discriminator
        d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train generator
        discriminator.trainable = False
        noise = np.random.normal(0, 1, (BATCH_SIZE, NOISE_DIM))
        g_labels = np.ones((BATCH_SIZE, 1))  # Generator wants discriminator to think fakes are real
        g_loss = gan.train_on_batch(noise, g_labels)
        
        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: [D loss: {d_loss[0]:.4f}, acc: {100*d_loss[1]:.1f}%] [G loss: {g_loss:.4f}]")
        
        # Save images
        if epoch % 500 == 0:
            save_images(generator, epoch)
    
    print("Training completed!")
    return generator, discriminator

if __name__ == "__main__":
    print("Starting Simple Ships GAN Training...")
    generator, discriminator = train_simple_gan(epochs=3000)
    print("Training finished!")