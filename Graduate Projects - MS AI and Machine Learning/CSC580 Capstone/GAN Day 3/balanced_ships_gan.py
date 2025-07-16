#!/usr/bin/env python3
"""
Balanced CIFAR10 Ships DCGAN - Fixed Discriminator Training
Properly balanced discriminator vs generator training with target 60-70% accuracy
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
    """Build DCGAN generator"""
    noise = Input(shape=(LATENT_DIM,))
    
    x = Dense(4 * 4 * 512, use_bias=False)(noise)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Reshape((4, 4, 512))(x)
    
    x = Conv2DTranspose(256, 4, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    x = Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    x = Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    img = Conv2DTranspose(3, 4, strides=1, padding='same', activation='tanh')(x)
    
    return Model(noise, img, name="generator")

def build_discriminator():
    """Build DCGAN discriminator"""
    img = Input(shape=IMG_SHAPE)
    
    x = Conv2D(64, 4, strides=2, padding='same')(img)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    
    x = Conv2D(128, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    
    x = Conv2D(256, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    
    x = Conv2D(512, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    
    x = Flatten()(x)
    validity = Dense(1, activation='sigmoid')(x)
    
    return Model(img, validity, name="discriminator")

def load_ships_data():
    """Load CIFAR10 ship images"""
    (X, y), (_, _) = cifar10.load_data()
    X_ships = X[y.flatten() == SHIPS_CLASS]
    X_ships = X_ships.astype('float32') / 127.5 - 1.0
    print(f"Loaded {X_ships.shape[0]} ship images")
    return X_ships

def calculate_accuracy(y_true, y_pred):
    """Calculate accuracy manually"""
    predictions = (y_pred > 0.5).astype(float)
    return np.mean(predictions == y_true)

def save_sample_images(generator, epoch, rows=4, cols=4):
    """Generate and save sample images"""
    noise = np.random.normal(0, 1, (rows * cols, LATENT_DIM))
    gen_imgs = generator.predict(noise, verbose=0)
    
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

def train_balanced_gan(epochs=15000, batch_size=32, sample_interval=1000, save_interval=5000):
    """Train balanced DCGAN with proper discriminator accuracy"""
    
    # Load data
    X_ships = load_ships_data()
    
    # Build models
    generator = build_generator()
    discriminator = build_discriminator()
    
    # Optimizers
    d_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    g_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    
    # Compile discriminator (without metrics to avoid confusion)
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optimizer)
    
    # Build combined model
    noise = Input(shape=(LATENT_DIM,))
    gen_img = generator(noise)
    discriminator.trainable = False
    validity = discriminator(gen_img)
    combined = Model(noise, validity, name="combined")
    combined.compile(loss='binary_crossentropy', optimizer=g_optimizer)
    
    print(f"🏗️  Generator parameters: {generator.count_params():,}")
    print(f"🏗️  Discriminator parameters: {discriminator.count_params():,}")
    print(f"🚀 Starting balanced training for {epochs} epochs...")
    
    # Training history
    d_losses = []
    g_losses = []
    d_accuracies = []
    
    start_time = datetime.now()
    
    for epoch in range(epochs):
        
        # Determine training strategy based on discriminator performance
        if epoch < 100:
            d_steps = 1  # Initial phase
        else:
            # Check recent discriminator accuracy
            recent_acc = d_accuracies[-10:] if len(d_accuracies) >= 10 else d_accuracies
            avg_acc = np.mean(recent_acc) if recent_acc else 0.5
            
            if avg_acc < 0.5:  # D too weak, train more
                d_steps = 3
            elif avg_acc > 0.8:  # D too strong, train less
                d_steps = 1
            else:  # In good range
                d_steps = 2
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        discriminator.trainable = True
        
        d_loss_total = 0
        d_acc_total = 0
        
        for _ in range(d_steps):
            # Real images
            idx = np.random.randint(0, X_ships.shape[0], batch_size)
            real_imgs = X_ships[idx]
            real_labels = np.ones((batch_size, 1))
            
            # Fake images
            noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
            fake_imgs = generator.predict(noise, verbose=0)
            fake_labels = np.zeros((batch_size, 1))
            
            # Train on real and fake
            d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)
            d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            
            # Calculate accuracy manually
            real_pred = discriminator.predict(real_imgs, verbose=0)
            fake_pred = discriminator.predict(fake_imgs, verbose=0)
            
            real_acc = calculate_accuracy(real_labels, real_pred)
            fake_acc = calculate_accuracy(fake_labels, fake_pred)
            d_acc = 0.5 * (real_acc + fake_acc)
            
            d_loss_total += d_loss
            d_acc_total += d_acc
        
        d_loss_avg = d_loss_total / d_steps
        d_acc_avg = d_acc_total / d_steps
        
        # -----------------
        #  Train Generator
        # -----------------
        
        discriminator.trainable = False
        
        noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
        g_labels = np.ones((batch_size, 1))  # Generator wants D to think fakes are real
        g_loss = combined.train_on_batch(noise, g_labels)
        
        # Store metrics
        d_losses.append(d_loss_avg)
        g_losses.append(g_loss)
        d_accuracies.append(d_acc_avg)
        
        # Print progress
        if epoch % 100 == 0:
            elapsed = datetime.now() - start_time
            print(f"{epoch:6d} [D loss: {d_loss_avg:.4f}, acc: {100*d_acc_avg:.1f}%] "
                  f"[G loss: {g_loss:.4f}] [D steps: {d_steps}] [{elapsed}]")
        
        # Save sample images
        if epoch % sample_interval == 0:
            save_sample_images(generator, epoch)
        
        # Save model weights
        if epoch % save_interval == 0 and epoch > 0:
            generator.save_weights(f"weights/generator_balanced_{epoch}.weights.h5")
            discriminator.save_weights(f"weights/discriminator_balanced_{epoch}.weights.h5")
            print(f"💾 Saved model weights for epoch {epoch}")
    
    print(f"✅ Training completed in {datetime.now() - start_time}")
    
    # Save final models
    generator.save_weights(f"weights/generator_balanced_final.weights.h5")
    discriminator.save_weights(f"weights/discriminator_balanced_final.weights.h5")
    
    # Plot training history
    plot_training_history(d_losses, g_losses, d_accuracies)
    
    return generator, discriminator

def plot_training_history(d_losses, g_losses, d_accuracies):
    """Plot comprehensive training history"""
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
    plt.fill_between(range(len(acc_percent)), 60, 70, alpha=0.2, color='orange')
    plt.title('Discriminator Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    # Smoothed metrics
    plt.subplot(1, 3, 3)
    window = min(100, len(d_losses) // 4)
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
    plt.savefig('images/balanced_training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("📈 Saved balanced training history plot")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train Balanced CIFAR10 Ships DCGAN')
    parser.add_argument('--epochs', type=int, default=15000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--sample_interval', type=int, default=1000, help='Sample interval')
    parser.add_argument('--save_interval', type=int, default=5000, help='Save interval')
    
    args = parser.parse_args()
    
    print("🚢 Balanced CIFAR10 Ships DCGAN")
    print("=" * 50)
    print(f"Target: Discriminator accuracy 60-70%")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 50)
    
    train_balanced_gan(
        epochs=args.epochs,
        batch_size=args.batch_size,
        sample_interval=args.sample_interval,
        save_interval=args.save_interval
    )

if __name__ == "__main__":
    main()