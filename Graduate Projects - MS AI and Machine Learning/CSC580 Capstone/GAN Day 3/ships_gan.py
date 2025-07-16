"""
CIFAR10 Ships DCGAN - Final Production Version
Generates realistic ship images using Deep Convolutional GAN
Category 8 from CIFAR10 dataset (ships)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Reshape, Flatten,
                                     Dropout, BatchNormalization, LeakyReLU,
                                     Conv2D, Conv2DTranspose)
from tensorflow.keras.optimizers import Adam
import argparse
from datetime import datetime

# Configuration
IMG_SHAPE = (32, 32, 3)
LATENT_DIM = 100
SHIPS_CLASS = 8

# Create directories
os.makedirs("images", exist_ok=True)
os.makedirs("weights", exist_ok=True)

def build_generator():
    """Build DCGAN generator for 32x32 CIFAR10 ship images"""
    
    noise = Input(shape=(LATENT_DIM,))
    
    # Project noise to 4x4x512 feature map
    x = Dense(4 * 4 * 512, use_bias=False)(noise)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Reshape((4, 4, 512))(x)
    
    # Upsample to 8x8x256
    x = Conv2DTranspose(256, 4, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    # Upsample to 16x16x128
    x = Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    # Upsample to 32x32x64
    x = Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    # Final layer to RGB (32x32x3)
    img = Conv2DTranspose(3, 4, strides=1, padding='same', activation='tanh')(x)
    
    return Model(noise, img, name="generator")

def build_discriminator():
    """Build DCGAN discriminator for 32x32 CIFAR10 ship images"""
    
    img = Input(shape=IMG_SHAPE)
    
    # 32x32x3 -> 16x16x64
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
    
    # Flatten and classify
    x = Flatten()(x)
    validity = Dense(1, activation='sigmoid')(x)
    
    return Model(img, validity, name="discriminator")

def load_ships_data():
    """Load and preprocess CIFAR10 ship images (class 8)"""
    (X, y), (_, _) = cifar10.load_data()
    
    # Filter for ships (class 8)
    X_ships = X[y.flatten() == SHIPS_CLASS]
    
    # Normalize to [-1, 1] for tanh activation
    X_ships = X_ships.astype('float32') / 127.5 - 1.0
    
    print(f"Loaded {X_ships.shape[0]} ship images")
    return X_ships

def save_sample_images(generator, epoch, rows=4, cols=4):
    """Generate and save sample images"""
    noise = np.random.normal(0, 1, (rows * cols, LATENT_DIM))
    gen_imgs = generator.predict(noise, verbose=0)
    
    # Rescale to [0, 1]
    gen_imgs = 0.5 * gen_imgs + 0.5
    gen_imgs = np.clip(gen_imgs, 0, 1)
    
    fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
    axes = axes.flatten()
    
    for i, img in enumerate(gen_imgs):
        axes[i].imshow(img)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"images/ships_epoch_{epoch}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"💾 Saved sample images for epoch {epoch}")

class ShipsGAN:
    """DCGAN for generating CIFAR10 ship images"""
    
    def __init__(self, learning_rate=0.0002, beta_1=0.5):
        self.lr = learning_rate
        self.beta_1 = beta_1
        
        # Build and compile models
        self.generator = build_generator()
        self.discriminator = build_discriminator()
        
        # Compile discriminator
        d_optimizer = Adam(learning_rate=self.lr, beta_1=self.beta_1)
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=d_optimizer,
            metrics=['accuracy']
        )
        
        # Build combined model for generator training
        noise = Input(shape=(LATENT_DIM,))
        gen_img = self.generator(noise)
        
        # Freeze discriminator for combined model
        self.discriminator.trainable = False
        validity = self.discriminator(gen_img)
        
        g_optimizer = Adam(learning_rate=self.lr, beta_1=self.beta_1)
        self.combined = Model(noise, validity, name="combined")
        self.combined.compile(loss='binary_crossentropy', optimizer=g_optimizer)
        
        print(f"🏗️  Generator parameters: {self.generator.count_params():,}")
        print(f"🏗️  Discriminator parameters: {self.discriminator.count_params():,}")
    
    def train(self, epochs=15000, batch_size=32, sample_interval=1000, save_interval=5000):
        """Train the DCGAN"""
        
        # Load data
        X_ships = load_ships_data()
        
        # Labels with label smoothing
        valid_smooth = np.ones((batch_size, 1)) * 0.9
        fake_smooth = np.zeros((batch_size, 1)) + 0.1
        valid_gen = np.ones((batch_size, 1))
        
        print(f"🚀 Starting training for {epochs} epochs...")
        start_time = datetime.now()
        
        # Training history
        d_losses = []
        g_losses = []
        
        for epoch in range(epochs):
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            self.discriminator.trainable = True
            
            # Train discriminator multiple times if it's performing poorly
            d_train_steps = 1
            if epoch > 100:  # After initial phase
                # If discriminator accuracy is too low, train it more
                recent_acc = d_losses[-10:] if len(d_losses) > 10 else [0.5]
                avg_acc = np.mean([loss[1] if isinstance(loss, (list, tuple)) else 0.5 for loss in recent_acc])
                if avg_acc < 0.4:  # If accuracy below 40%
                    d_train_steps = 3
                elif avg_acc > 0.8:  # If accuracy above 80%
                    d_train_steps = 1
                else:
                    d_train_steps = 2
            
            d_loss_total = [0, 0]
            for _ in range(d_train_steps):
                # Sample real images
                idx = np.random.randint(0, X_ships.shape[0], batch_size)
                real_imgs = X_ships[idx]
                
                # Generate fake images
                noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
                fake_imgs = self.generator.predict(noise, verbose=0)
                
                # Train discriminator with stronger labels
                valid_strong = np.ones((batch_size, 1)) * 0.95  # Stronger real labels
                fake_strong = np.zeros((batch_size, 1)) + 0.05  # Stronger fake labels
                
                d_loss_real = self.discriminator.train_on_batch(real_imgs, valid_strong)
                d_loss_fake = self.discriminator.train_on_batch(fake_imgs, fake_strong)
                
                d_loss_step = 0.5 * np.add(d_loss_real, d_loss_fake)
                d_loss_total = np.add(d_loss_total, d_loss_step)
            
            d_loss = d_loss_total / d_train_steps
            
            # -----------------
            #  Train Generator
            # -----------------
            
            self.discriminator.trainable = False
            
            # Train generator (only once per epoch to prevent it from overwhelming discriminator)
            noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
            g_loss = self.combined.train_on_batch(noise, valid_gen)
            
            # Store losses (store full d_loss for accuracy tracking)
            d_losses.append(d_loss)
            g_losses.append(g_loss)
            
            # Print progress with training balance info
            if epoch % 100 == 0:
                elapsed = datetime.now() - start_time
                print(f"{epoch:6d} [D loss: {d_loss[0]:.4f}, acc: {100*d_loss[1]:.1f}%] "
                      f"[G loss: {g_loss:.4f}] [D steps: {d_train_steps}] [{elapsed}]")
            
            # Save sample images
            if epoch % sample_interval == 0:
                save_sample_images(self.generator, epoch)
            
            # Save model weights
            if epoch % save_interval == 0 and epoch > 0:
                self.save_models(epoch)
                
        print(f"✅ Training completed in {datetime.now() - start_time}")
        
        # Save final models
        self.save_models(epochs)
        
        # Plot training history
        self.plot_training_history(d_losses, g_losses)
    
    def save_models(self, epoch):
        """Save model weights"""
        self.generator.save_weights(f"weights/generator_ships_{epoch}.weights.h5")
        self.discriminator.save_weights(f"weights/discriminator_ships_{epoch}.weights.h5")
        print(f"💾 Saved model weights for epoch {epoch}")
    
    def plot_training_history(self, d_losses, g_losses):
        """Plot training loss history with accuracy"""
        plt.figure(figsize=(15, 5))
        
        # Extract discriminator losses and accuracies
        d_loss_vals = [loss[0] if isinstance(loss, (list, tuple)) else loss for loss in d_losses]
        d_acc_vals = [loss[1] if isinstance(loss, (list, tuple)) else 0.5 for loss in d_losses]
        
        plt.subplot(1, 3, 1)
        plt.plot(d_loss_vals, label='Discriminator Loss', alpha=0.7, color='red')
        plt.plot(g_losses, label='Generator Loss', alpha=0.7, color='blue')
        plt.title('Training Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot([acc * 100 for acc in d_acc_vals], label='Discriminator Accuracy', alpha=0.8, color='green')
        plt.axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='Target Range')
        plt.axhline(y=70, color='orange', linestyle='--', alpha=0.7)
        plt.title('Discriminator Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        
        plt.subplot(1, 3, 3)
        # Smooth the losses with moving average
        window = 100
        if len(d_loss_vals) > window:
            d_smooth = np.convolve(d_loss_vals, np.ones(window)/window, mode='valid')
            g_smooth = np.convolve(g_losses, np.ones(window)/window, mode='valid')
            acc_smooth = np.convolve([acc * 100 for acc in d_acc_vals], np.ones(window)/window, mode='valid')
            plt.plot(d_smooth, label='D Loss (smooth)', alpha=0.8, color='red')
            plt.plot(g_smooth, label='G Loss (smooth)', alpha=0.8, color='blue')
            plt.plot(acc_smooth, label='D Acc (smooth)', alpha=0.8, color='green')
        plt.title('Smoothed Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('images/training_history.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("📈 Saved training history plot")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train CIFAR10 Ships DCGAN')
    parser.add_argument('--epochs', type=int, default=15000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--sample_interval', type=int, default=1000, help='Sample image save interval')
    parser.add_argument('--save_interval', type=int, default=5000, help='Model save interval')
    
    args = parser.parse_args()
    
    print("🚢 CIFAR10 Ships DCGAN")
    print("=" * 50)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("=" * 50)
    
    # Create and train GAN
    gan = ShipsGAN(learning_rate=args.lr)
    gan.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        sample_interval=args.sample_interval,
        save_interval=args.save_interval
    )

if __name__ == "__main__":
    main()