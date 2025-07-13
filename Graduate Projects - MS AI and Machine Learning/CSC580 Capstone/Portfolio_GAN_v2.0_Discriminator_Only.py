import numpy as np
from keras.models import Sequential, Model
from keras.layers import (
    Input, Dense, Flatten, Dropout,
    BatchNormalization, LeakyReLU,
    Conv2D, ZeroPadding2D
)

# Parameters to match your setup
image_shape = (32, 32, 3)


def build_discriminator():
    """
    Discriminator inspired by assignment requirements but improved

    Assignment uses:
    - Conv2D layers with 3x3 kernels
    - LeakyReLU activations
    - Dropout for regularization
    - BatchNormalization
    - ZeroPadding2D for size matching

    Our improvements:
    - Better filter progression
    - Consistent activation parameters
    - Optimized layer order
    """
    model = Sequential()

    # First conv block: 32x32 -> 16x16
    # Assignment: Conv2D(32, kernel_size=3, strides=2, ...)
    model.add(Conv2D(64, kernel_size=4, strides=2,
                     input_shape=image_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    # Second conv block: 16x16 -> 8x8
    # Assignment uses ZeroPadding2D here - we'll include it for compatibility
    model.add(Conv2D(128, kernel_size=4, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))  # Following assignment pattern
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    # Third conv block: 8x8 -> 4x4
    # Assignment: Conv2D(128, kernel_size=3, strides=2, ...)
    model.add(Conv2D(256, kernel_size=4, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    # Fourth conv block: 4x4 -> 2x2
    # Assignment: Conv2D(256, kernel_size=3, strides=1, ...)
    model.add(Conv2D(512, kernel_size=4, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    # Output layer
    # Assignment: Flatten() -> Dense(1, activation='sigmoid')
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    # Create the model with input/output
    image = Input(shape=image_shape)
    validity = model(image)

    return Model(image, validity)


# Test the discriminator
if __name__ == "__main__":
    print("Building discriminator...")
    discriminator = build_discriminator()

    print("\nDiscriminator Architecture:")
    discriminator.summary()

    print(f"\nTotal parameters: {discriminator.count_params():,}")

    # Test with sample data
    print("\nTesting with sample ship images...")
    sample_images = np.random.normal(0, 1, (5, 32, 32, 3))  # Fake sample
    predictions = discriminator.predict(sample_images, verbose=0)

    print("Sample predictions (closer to 1 = more 'real'):")
    for i, pred in enumerate(predictions):
        print(f"  Image {i + 1}: {pred[0]:.4f}")

    print("\nDiscriminator ready to pair with your proven generator!")