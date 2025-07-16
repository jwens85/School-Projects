#!/usr/bin/env python3

import numpy as np
from cifar10_ships_gan import build_generator, build_discriminator, load_ships_data

def test_models():
    """Test that the GAN models build and work correctly"""
    print("Testing CIFAR10 Ships GAN models...")
    
    # Test data loading
    print("\n1. Testing data loading...")
    ships_data = load_ships_data()
    print(f"✓ Loaded {ships_data.shape[0]} ship images")
    print(f"✓ Image shape: {ships_data.shape[1:]}")
    print(f"✓ Data range: [{ships_data.min():.2f}, {ships_data.max():.2f}]")
    
    # Test generator
    print("\n2. Testing generator...")
    generator = build_generator()
    print(f"✓ Generator built successfully")
    print(f"✓ Generator input shape: {generator.input_shape}")
    print(f"✓ Generator output shape: {generator.output_shape}")
    
    # Test generator inference
    noise = np.random.normal(0, 1, (1, 100))
    fake_image = generator.predict(noise, verbose=0)
    print(f"✓ Generated image shape: {fake_image.shape}")
    print(f"✓ Generated image range: [{fake_image.min():.2f}, {fake_image.max():.2f}]")
    
    # Test discriminator
    print("\n3. Testing discriminator...")
    discriminator = build_discriminator()
    print(f"✓ Discriminator built successfully")
    print(f"✓ Discriminator input shape: {discriminator.input_shape}")
    print(f"✓ Discriminator output shape: {discriminator.output_shape}")
    
    # Test discriminator inference
    real_batch = ships_data[:1]
    real_pred = discriminator.predict(real_batch, verbose=0)
    fake_pred = discriminator.predict(fake_image, verbose=0)
    print(f"✓ Real image prediction: {real_pred[0][0]:.4f}")
    print(f"✓ Fake image prediction: {fake_pred[0][0]:.4f}")
    
    print("\n🎉 All tests passed! The CIFAR10 Ships GAN is ready to train.")
    print("\nTo start training, run:")
    print("python3 cifar10_ships_gan.py")

if __name__ == "__main__":
    test_models()