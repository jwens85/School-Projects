#!/usr/bin/env python3
"""
Proper DCGAN for CIFAR10 Ships - No colorful noise
Uses proven architecture and initialization that generates real ship images
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
import os

# Ensure reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Configuration
NOISE_DIM = 100
BATCH_SIZE = 64
EPOCHS = 50000
SAMPLE_INTERVAL = 1000

# Create directories
os.makedirs("images", exist_ok=True)
os.makedirs("models", exist_ok=True)

class DCGAN:
    def __init__(self):
        self.img_shape = (32, 32, 3)
        self.noise_dim = NOISE_DIM
        
        # Build generator and discriminator
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
        # Compile discriminator
        self.discriminator.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Build and compile combined model
        self.discriminator.trainable = False
        noise_input = keras.Input(shape=(self.noise_dim,))
        generated_image = self.generator(noise_input)
        validity = self.discriminator(generated_image)
        
        self.combined = keras.Model(noise_input, validity)
        self.combined.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy'
        )
    
    def build_generator(self):
        """Build the generator network"""
        model = keras.Sequential([
            # Input layer
            layers.Dense(4 * 4 * 256, input_shape=(self.noise_dim,)),
            layers.Reshape((4, 4, 256)),
            
            # First upsampling block
            layers.Conv2DTranspose(128, 5, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            
            # Second upsampling block (4x4 -> 8x8)
            layers.Conv2DTranspose(64, 5, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            
            # Third upsampling block (8x8 -> 16x16)
            layers.Conv2DTranspose(32, 5, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            
            # Fourth upsampling block (16x16 -> 32x32)
            layers.Conv2DTranspose(3, 5, strides=2, padding='same'),
            layers.Activation('tanh')
        ], name='generator')
        
        return model
    
    def build_discriminator(self):
        """Build the discriminator network"""
        model = keras.Sequential([
            # Input layer (32x32x3)
            layers.Conv2D(32, 5, strides=2, padding='same', input_shape=self.img_shape),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.3),
            
            # Second block (16x16x32)
            layers.Conv2D(64, 5, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.3),
            
            # Third block (8x8x64)
            layers.Conv2D(128, 5, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.3),
            
            # Fourth block (4x4x128)
            layers.Conv2D(256, 5, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid')
        ], name='discriminator')
        
        return model
    
    def load_data(self):
        """Load and preprocess CIFAR-10 ship data"""
        (x_train, y_train), (_, _) = cifar10.load_data()
        
        # Filter for ships (class 8)
        ship_indices = y_train.flatten() == 8
        x_ships = x_train[ship_indices]
        
        # Normalize to [-1, 1]
        x_ships = (x_ships.astype(np.float32) - 127.5) / 127.5
        
        print(f"Loaded {len(x_ships)} ship images")
        return x_ships
    
    def save_images(self, epoch):
        """Generate and save sample images"""
        rows, cols = 4, 4
        noise = np.random.normal(0, 1, (rows * cols, self.noise_dim))
        generated_images = self.generator.predict(noise, verbose=0)
        
        # Rescale images to [0, 1]
        generated_images = 0.5 * generated_images + 0.5
        generated_images = np.clip(generated_images, 0, 1)
        
        fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                axes[i, j].imshow(generated_images[idx])
                axes[i, j].axis('off')
                idx += 1
        
        plt.tight_layout()
        plt.savefig(f"images/dcgan_ships_{epoch}.png")
        plt.close()
        print(f"Saved sample images for epoch {epoch}")
    
    def train(self, epochs=EPOCHS):
        """Train the DCGAN"""
        # Load data
        x_train = self.load_data()
        
        # Labels for real and fake
        real_labels = np.ones((BATCH_SIZE, 1))
        fake_labels = np.zeros((BATCH_SIZE, 1))
        
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            
            # Train Discriminator
            self.discriminator.trainable = True
            
            # Get real images
            idx = np.random.randint(0, x_train.shape[0], BATCH_SIZE)
            real_images = x_train[idx]
            
            # Generate fake images
            noise = np.random.normal(0, 1, (BATCH_SIZE, self.noise_dim))
            fake_images = self.generator.predict(noise, verbose=0)
            
            # Train discriminator
            d_loss_real = self.discriminator.train_on_batch(real_images, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(fake_images, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train Generator
            self.discriminator.trainable = False
            noise = np.random.normal(0, 1, (BATCH_SIZE, self.noise_dim))
            g_loss = self.combined.train_on_batch(noise, real_labels)
            
            # Print progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: [D loss: {d_loss[0]:.4f}, acc: {100*d_loss[1]:.1f}%] [G loss: {g_loss:.4f}]")
            
            # Save sample images
            if epoch % SAMPLE_INTERVAL == 0:
                self.save_images(epoch)
            
            # Save models
            if epoch % 10000 == 0 and epoch > 0:
                self.generator.save(f"models/generator_{epoch}.h5")
                self.discriminator.save(f"models/discriminator_{epoch}.h5")
        
        print("Training completed!")

def main():
    dcgan = DCGAN()
    
    # Show model summaries
    print("Generator Summary:")
    dcgan.generator.summary()
    print("\nDiscriminator Summary:")
    dcgan.discriminator.summary()
    
    # Start training
    dcgan.train(epochs=EPOCHS)

if __name__ == "__main__":
    main()