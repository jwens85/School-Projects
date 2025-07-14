import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import (
    Conv2D, LeakyReLU, Dropout, BatchNormalization,
    ZeroPadding2D, Flatten, Dense
)
from keras.optimizers import Adam

print("Discriminator v2.1 - Ship vs Noise Test")

# Load CIFAR-10 ships
print("Loading CIFAR-10 ship data...")
(X, y), (_, _) = cifar10.load_data()
X_ships = X[y.flatten() == 8]  # Ships only
print(f"Loaded {len(X_ships)} ship images")

# Normalize to [-1, 1]
X_ships = (X_ships.astype(np.float32) - 127.5) / 127.5

# Split ships into train/test
split_idx = len(X_ships) // 2
train_ships = X_ships[:split_idx]
test_ships = X_ships[split_idx:]
print(f"Train ships: {len(train_ships)}, Test ships: {len(test_ships)}")


def build_discriminator():
    """Basic discriminator - can it tell ships from noise?"""
    model = Sequential()

    # 32x32 -> 16x16
    model.add(Conv2D(64, kernel_size=4, strides=2, input_shape=(32, 32, 3), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    # 16x16 -> 8x8
    model.add(Conv2D(128, kernel_size=4, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    # 8x8 -> 4x4
    model.add(Conv2D(256, kernel_size=4, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    # 4x4 -> 2x2
    model.add(Conv2D(512, kernel_size=4, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    # Output
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model


# Build and compile discriminator
print("\nBuilding discriminator...")
discriminator = build_discriminator()
discriminator.compile(
    optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(f"Discriminator parameters: {discriminator.count_params():,}")

# Create training data: ships vs noise
print("\nCreating training data...")
n_train = len(train_ships)

# Generate random noise images
train_noise = np.random.normal(0, 1, (n_train, 32, 32, 3))

# Combine ships and noise
X_train = np.vstack([train_ships, train_noise])
y_train = np.vstack([np.ones((n_train, 1)), np.zeros((n_train, 1))])

# Shuffle training data
indices = np.random.permutation(len(X_train))
X_train = X_train[indices]
y_train = y_train[indices]

print(f"Training data: {len(X_train)} images (50% ships, 50% noise)")

# Create test data
print("Creating test data...")
n_test = len(test_ships)
test_noise = np.random.normal(0, 1, (n_test, 32, 32, 3))

X_test = np.vstack([test_ships, test_noise])
y_test = np.vstack([np.ones((n_test, 1)), np.zeros((n_test, 1))])

print(f"Test data: {len(X_test)} images (50% ships, 50% noise)")

# Train discriminator
print("\nTraining discriminator on ships vs noise...")
print("Target: >90% accuracy to prove it can distinguish them")

history = discriminator.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Evaluate final performance
print("\n" + "-" * 50)
print("FINAL EVALUATION")
print("-" * 50)

train_loss, train_acc = discriminator.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = discriminator.evaluate(X_test, y_test, verbose=0)

print(f"Training Accuracy: {train_acc:.4f} ({train_acc * 100:.1f}%)")
print(f"Test Accuracy: {test_acc:.4f} ({test_acc * 100:.1f}%)")

# Test on specific examples
print("\nTesting on specific examples...")
ship_predictions = discriminator.predict(test_ships[:5], verbose=0)
noise_predictions = discriminator.predict(test_noise[:5], verbose=0)

print("Ship predictions (should be close to 1.0):")
for i, pred in enumerate(ship_predictions):
    print(f"  Ship {i + 1}: {pred[0]:.4f}")

print("Noise predictions (should be close to 0.0):")
for i, pred in enumerate(noise_predictions):
    print(f"  Noise {i + 1}: {pred[0]:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Discriminator Accuracy: Ships vs Noise')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Discriminator Loss: Ships vs Noise')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('discriminator_v21_test_results.png', dpi=150, bbox_inches='tight')
plt.show()

# Conclusion
print("\n" + "-" * 50)
print("CONCLUSION")
print("-" * 50)

if test_acc > 0.9:
    print("SUCCESS: Discriminator can distinguish ships from noise!")
    print("   - Architecture is working correctly")
    print("   - Data preprocessing is good")
    print("   - Ready for GAN training with pre-trained weights")

    # Save the trained discriminator
    discriminator.save('discriminator_v21_pretrained.keras')
    print("   - Saved trained discriminator as 'discriminator_v21_pretrained.keras'")

elif test_acc > 0.7:
    print("PARTIAL: Discriminator shows some learning but not great")
    print("   - May need architecture adjustments")
    print("   - Or more training epochs")

else:
    print("FAILURE: Discriminator cannot distinguish ships from noise")
    print("   - Architecture may be fundamentally broken")
    print("   - Or data preprocessing issue")
    print("   - Need to debug further")

print(f"\nFinal test accuracy: {test_acc * 100:.1f}%")