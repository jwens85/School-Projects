#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from cifar10_ships_gan import build_generator, build_discriminator, load_ships_data, train_gan

def quick_test():
    """Quick test of the improved GAN implementation"""
    print("🚢 Testing Improved CIFAR10 Ships GAN")
    print("=" * 50)
    
    # Test data loading
    print("📊 Loading ship data...")
    ships_data = load_ships_data()
    print(f"✅ Loaded {ships_data.shape[0]} ship images")
    
    # Test models
    print("\n🏗️  Building models...")
    generator = build_generator()
    discriminator = build_discriminator()
    print("✅ Models built successfully")
    
    # Print model summaries
    print(f"\n📋 Generator: {generator.count_params():,} parameters")
    print(f"📋 Discriminator: {discriminator.count_params():,} parameters")
    
    # Test forward pass
    print("\n🔄 Testing forward pass...")
    noise = np.random.normal(0, 1, (4, 100))
    fake_imgs = generator.predict(noise, verbose=0)
    
    real_batch = ships_data[:4]
    real_pred = discriminator.predict(real_batch, verbose=0)
    fake_pred = discriminator.predict(fake_imgs, verbose=0)
    
    print(f"Generated images shape: {fake_imgs.shape}")
    print(f"Generated images range: [{fake_imgs.min():.3f}, {fake_imgs.max():.3f}]")
    print(f"Real predictions mean: {real_pred.mean():.4f}")
    print(f"Fake predictions mean: {fake_pred.mean():.4f}")
    
    print("\n🎯 Running mini training test (10 epochs)...")
    train_gan(epochs=10, batch_size=16, sample_interval=5)
    
    print("\n🎉 All tests completed successfully!")
    print("\n🚀 To start full training, run:")
    print("   python3 cifar10_ships_gan.py")

if __name__ == "__main__":
    quick_test()