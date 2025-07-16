#!/usr/bin/env python3
"""
CIFAR10 Ships GAN v0.2 - Fixed Architecture
Addresses fundamental issues from v0.1:
- Proper discriminator training (60-70% accuracy)
- Stable architecture with correct layer dimensions
- Proper loss scaling and weight initialization
- Outputs to images/0.2/ subfolder
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose, Conv2D, LeakyReLU, BatchNormalization, Dropout, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

# Configuration
IMG_SHAPE = (32, 32, 3)
NOISE_DIM = 100
SHIPS_CLASS = 8

# Create output directory
os.makedirs("images/0.2", exist_ok=True)
os.makedirs("weights", exist_ok=True)

def load_ships_data():
    """Load and preprocess CIFAR10 ship images"""
    (X, y), (_, _) = cifar10.load_data()
    X_ships = X[y.flatten() == SHIPS_CLASS]
    X_ships = X_ships.astype('float32') / 127.5 - 1.0  # Normalize to [-1, 1]
    print(f"Loaded {X_ships.shape[0]} ship images")
    return X_ships

def build_generator():
    """Build generator with proper weight initialization"""
    init = RandomNormal(stddev=0.02)
    
    noise = Input(shape=(NOISE_DIM,))
    
    # Dense layer to 4x4x512
    x = Dense(4 * 4 * 512, kernel_initializer=init)(noise)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Reshape((4, 4, 512))(x)
    
    # Upsample to 8x8x256
    x = Conv2DTranspose(256, 4, strides=2, padding='same', kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    # Upsample to 16x16x128
    x = Conv2DTranspose(128, 4, strides=2, padding='same', kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    # Upsample to 32x32x64
    x = Conv2DTranspose(64, 4, strides=2, padding='same', kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    # Final layer to RGB
    img = Conv2DTranspose(3, 4, strides=1, padding='same', activation='tanh', kernel_initializer=init)(x)
    
    return Model(noise, img, name="generator")

def build_discriminator():
    """Build discriminator with proper capacity"""
    init = RandomNormal(stddev=0.02)
    
    img = Input(shape=IMG_SHAPE)
    
    # 32x32x3 -> 16x16x64
    x = Conv2D(64, 4, strides=2, padding='same', kernel_initializer=init)(img)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    
    # 16x16x64 -> 8x8x128
    x = Conv2D(128, 4, strides=2, padding='same', kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    
    # 8x8x128 -> 4x4x256
    x = Conv2D(256, 4, strides=2, padding='same', kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    
    # 4x4x256 -> 2x2x512
    x = Conv2D(512, 4, strides=2, padding='same', kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    
    # Output
    x = Flatten()(x)
    validity = Dense(1, activation='sigmoid', kernel_initializer=init)(x)
    
    return Model(img, validity, name="discriminator")

def calculate_accuracy(y_true, y_pred):
    """Calculate classification accuracy"""
    predictions = (y_pred > 0.5).astype(float)
    return np.mean(predictions == y_true)

def save_sample_images(generator, epoch):
    """Save sample images to 0.2 subfolder"""
    rows, cols = 4, 4
    noise = np.random.normal(0, 1, (rows * cols, NOISE_DIM))
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
    plt.savefig(f"images/0.2/ships_epoch_{epoch}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"💾 Saved sample images for epoch {epoch}")

class ImprovedGAN:
    """Improved GAN with proper training balance"""
    
    def __init__(self, learning_rate=0.0002):
        self.lr = learning_rate
        
        # Build models
        self.generator = build_generator()
        self.discriminator = build_discriminator()
        
        # Optimizers with proper parameters
        self.d_optimizer = Adam(learning_rate=self.lr, beta_1=0.5, beta_2=0.999)
        self.g_optimizer = Adam(learning_rate=self.lr, beta_1=0.5, beta_2=0.999)
        
        # Compile discriminator
        self.discriminator.compile(
            optimizer=self.d_optimizer,
            loss='binary_crossentropy'
        )
        
        # Build combined model
        noise = Input(shape=(NOISE_DIM,))
        gen_img = self.generator(noise)
        
        # Freeze discriminator for generator training
        self.discriminator.trainable = False
        validity = self.discriminator(gen_img)
        
        self.combined = Model(noise, validity, name="combined")
        self.combined.compile(
            optimizer=self.g_optimizer,
            loss='binary_crossentropy'
        )
        
        print(f"🏗️  Generator parameters: {self.generator.count_params():,}")
        print(f"🏗️  Discriminator parameters: {self.discriminator.count_params():,}")
    
    def train(self, epochs=15000, batch_size=32, sample_interval=1000):
        """Train with proper discriminator balance"""
        
        # Load data
        X_ships = load_ships_data()
        
        print(f"🚀 Starting training for {epochs} epochs...")
        start_time = datetime.now()
        
        # Training history
        d_accuracies = []
        d_losses = []
        g_losses = []
        
        for epoch in range(epochs):
            
            # Determine discriminator training steps based on performance
            if epoch < 200:
                d_steps = 1  # Initial training
            else:
                # Adaptive training based on recent accuracy
                recent_acc = d_accuracies[-10:] if len(d_accuracies) >= 10 else d_accuracies
                avg_acc = np.mean(recent_acc) if recent_acc else 0.5
                
                if avg_acc < 0.4:  # D too weak
                    d_steps = 5  # Train D more
                elif avg_acc < 0.6:  # Getting better
                    d_steps = 3
                elif avg_acc > 0.8:  # D too strong
                    d_steps = 1
                else:  # Good range
                    d_steps = 2
            
            # Train Discriminator
            self.discriminator.trainable = True
            
            d_loss_total = 0
            d_acc_total = 0
            
            for _ in range(d_steps):
                # Real images
                idx = np.random.randint(0, X_ships.shape[0], batch_size)
                real_imgs = X_ships[idx]
                
                # Add noise to labels (label smoothing)
                real_labels = np.ones((batch_size, 1)) - 0.1 * np.random.random((batch_size, 1))
                
                # Generate fake images
                noise = np.random.normal(0, 1, (batch_size, NOISE_DIM))
                fake_imgs = self.generator.predict(noise, verbose=0)
                fake_labels = np.zeros((batch_size, 1)) + 0.1 * np.random.random((batch_size, 1))
                
                # Train discriminator
                d_loss_real = self.discriminator.train_on_batch(real_imgs, real_labels)
                d_loss_fake = self.discriminator.train_on_batch(fake_imgs, fake_labels)
                d_loss = 0.5 * (d_loss_real + d_loss_fake)
                
                # Calculate accuracy
                real_pred = self.discriminator.predict(real_imgs, verbose=0)
                fake_pred = self.discriminator.predict(fake_imgs, verbose=0)
                
                real_acc = calculate_accuracy(np.ones((batch_size, 1)), real_pred)
                fake_acc = calculate_accuracy(np.zeros((batch_size, 1)), fake_pred)
                d_acc = 0.5 * (real_acc + fake_acc)
                
                d_loss_total += d_loss
                d_acc_total += d_acc
            
            d_loss_avg = d_loss_total / d_steps
            d_acc_avg = d_acc_total / d_steps
            
            # Train Generator
            self.discriminator.trainable = False
            
            noise = np.random.normal(0, 1, (batch_size, NOISE_DIM))
            g_labels = np.ones((batch_size, 1))  # Generator wants D to think fakes are real
            g_loss = self.combined.train_on_batch(noise, g_labels)
            
            # Store metrics
            d_losses.append(d_loss_avg)
            g_losses.append(g_loss)
            d_accuracies.append(d_acc_avg)
            
            # Print progress
            if epoch % 100 == 0:
                elapsed = datetime.now() - start_time
                status = "🎯" if 0.6 <= d_acc_avg <= 0.7 else "⚠️"
                print(f"{epoch:6d} [D loss: {d_loss_avg:.4f}, acc: {100*d_acc_avg:.1f}%] "
                      f"[G loss: {g_loss:.4f}] [D steps: {d_steps}] [{elapsed}] {status}")
            
            # Save sample images
            if epoch % sample_interval == 0:
                save_sample_images(self.generator, epoch)
            
            # Save model weights
            if epoch % 5000 == 0 and epoch > 0:
                self.generator.save_weights(f"weights/generator_v0.2_{epoch}.weights.h5")
                self.discriminator.save_weights(f"weights/discriminator_v0.2_{epoch}.weights.h5")
        
        print(f"✅ Training completed in {datetime.now() - start_time}")
        
        # Save final models
        self.generator.save_weights("weights/generator_v0.2_final.weights.h5")
        self.discriminator.save_weights("weights/discriminator_v0.2_final.weights.h5")
        
        # Plot training history
        self.plot_training_history(d_losses, g_losses, d_accuracies)
        
        return d_accuracies
    
    def plot_training_history(self, d_losses, g_losses, d_accuracies):
        """Plot training metrics"""
        plt.figure(figsize=(15, 5))
        
        # Losses
        plt.subplot(1, 3, 1)
        plt.plot(d_losses, label='Discriminator Loss', alpha=0.7, color='red')
        plt.plot(g_losses, label='Generator Loss', alpha=0.7, color='blue')
        plt.title('Training Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Discriminator Accuracy
        plt.subplot(1, 3, 2)
        acc_percent = [acc * 100 for acc in d_accuracies]
        plt.plot(acc_percent, label='Discriminator Accuracy', color='green')
        plt.axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='Target: 60-70%')
        plt.axhline(y=70, color='orange', linestyle='--', alpha=0.7)
        plt.fill_between(range(len(acc_percent)), 60, 70, alpha=0.2, color='green')
        plt.title('Discriminator Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        
        # Smoothed metrics
        plt.subplot(1, 3, 3)
        window = min(100, len(d_losses) // 10)
        if window > 1:
            d_smooth = np.convolve(d_losses, np.ones(window)/window, mode='valid')
            g_smooth = np.convolve(g_losses, np.ones(window)/window, mode='valid')
            acc_smooth = np.convolve(acc_percent, np.ones(window)/window, mode='valid')
            
            plt.plot(d_smooth, label='D Loss (smooth)', color='red', alpha=0.8)
            plt.plot(g_smooth, label='G Loss (smooth)', color='blue', alpha=0.8)
            plt.plot(acc_smooth, label='D Acc % (smooth)', color='green', alpha=0.8)
        
        plt.title('Smoothed Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('images/0.2/training_history_v0.2.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("📈 Saved training history plot")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train CIFAR10 Ships GAN v0.2')
    parser.add_argument('--epochs', type=int, default=15000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--sample_interval', type=int, default=1000, help='Sample image save interval')
    
    args = parser.parse_args()
    
    print("🚢 CIFAR10 Ships GAN v0.2 - Fixed Architecture")
    print("=" * 60)
    print(f"Target: Discriminator accuracy 60-70%")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Output folder: images/0.2/")
    print("=" * 60)
    
    # Create and train GAN
    gan = ImprovedGAN(learning_rate=args.lr)
    accuracies = gan.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        sample_interval=args.sample_interval
    )
    
    # Final results
    final_acc = accuracies[-1] * 100
    print(f"\n📊 Final discriminator accuracy: {final_acc:.1f}%")
    
    if 60 <= final_acc <= 70:
        print("🎯 SUCCESS! Discriminator accuracy in target range (60-70%)")
    else:
        print("⚠️  Discriminator accuracy outside target range - may need more training")

if __name__ == "__main__":
    main()