import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from keras.datasets import cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, LeakyReLU
from keras.layers import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

# Define parameters
image_shape = (32, 32, 3)
latent_dimensions = 100

# Timestamped output directory
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
base_output_dir = os.path.join("outputs", "v0.3", timestamp)
os.makedirs(base_output_dir, exist_ok=True)

# Load CIFAR-10 data and select airplane class (class 0)
(X, y), (_, _) = cifar10.load_data()
X = X[y.flatten() == 0]  # Select airplane images
print(f"Loaded {len(X)} airplane images")

# Define generator following assignment structure
def build_generator():
    model = Sequential()
    
    # Building the input layer
    model.add(Dense(128 * 8 * 8, activation="relu", input_dim=latent_dimensions))
    model.add(Reshape((8, 8, 128)))
    
    # First upsampling block
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.78))
    model.add(Activation("relu"))
    
    # Second upsampling block  
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.78))
    model.add(Activation("relu"))
    
    # Output layer
    model.add(Conv2D(3, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))
    
    # Generating the output image
    noise = Input(shape=(latent_dimensions,))
    image = model(noise)
    return Model(noise, image)

# Define discriminator following assignment structure
def build_discriminator():
    # Building the convolutional layers to classify whether an image is real or fake
    model = Sequential()
    
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=image_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.82))
    model.add(LeakyReLU(alpha=0.25))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.82))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.25))
    model.add(Dropout(0.25))
    
    # Building the output layer
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    
    image = Input(shape=image_shape)
    validity = model(image)
    
    return Model(image, validity)

# Display function following assignment structure
def display_images(generator, epoch):
    r, c = 4, 4
    noise = np.random.normal(0, 1, (r * c, latent_dimensions))
    generated_images = generator.predict(noise, verbose=0)
    
    # Scaling the generated images
    generated_images = 0.5 * generated_images + 0.5
    
    fig, axs = plt.subplots(r, c, figsize=(8, 8))
    count = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(generated_images[count, :,:,:])
            axs[i,j].axis('off')
            count += 1
    
    plt.suptitle(f'Generated Airplane Images - Epoch {epoch}')
    plt.tight_layout()
    
    # Save the figure as a 4x4 grid
    filename = f"epoch_{epoch}_grid.png"
    filepath = os.path.join(base_output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=100)
    plt.close()

# Build and compile the discriminator
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy',
                     optimizer=Adam(0.0002, 0.5),
                     metrics=['accuracy'])

# Making the discriminator untrainable so that the generator can learn from fixed gradient
discriminator.trainable = False

# Building the generator
generator = build_generator()

# Defining the input for the generator and generating the images
z = Input(shape=(latent_dimensions,))
image = generator(z)

# Checking the validity of the generated image
valid = discriminator(image)

# Defining the combined model of the generator and the discriminator
combined_network = Model(z, valid)
combined_network.compile(loss='binary_crossentropy',
                        optimizer=Adam(0.0002, 0.5))

# Training parameters
num_epochs = 2000  # Reduced for testing, increase when working
batch_size = 32
display_interval = 500
losses = []

# Normalizing the input
X = (X / 127.5) - 1.

# Defining the Adversarial ground truths
valid_labels = np.ones((batch_size, 1))
# Adding some noise for stability
valid_labels += 0.05 * np.random.random(valid_labels.shape)

fake_labels = np.zeros((batch_size, 1))
fake_labels += 0.05 * np.random.random(fake_labels.shape)

print(f"Starting training for {num_epochs} epochs...")
print(f"Output directory: {base_output_dir}")

# Training loop following assignment structure
for epoch in range(num_epochs):
    
    # Training the Discriminator
    
    # Sampling a random batch of images
    index = np.random.randint(0, X.shape[0], batch_size)
    images = X[index]
    
    # Sampling noise and generating a batch of new images
    noise = np.random.normal(0, 1, (batch_size, latent_dimensions))
    generated_images = generator.predict(noise, verbose=0)
    
    # Training the discriminator to detect more accurately
    # whether a generated image is real or fake
    discm_loss_real = discriminator.train_on_batch(images, valid_labels)
    discm_loss_fake = discriminator.train_on_batch(generated_images, fake_labels)
    discm_loss = 0.5 * np.add(discm_loss_real, discm_loss_fake)
    
    # Training the generator
    
    # Generate new noise for generator training
    noise = np.random.normal(0, 1, (batch_size, latent_dimensions))
    
    # Training the generator to generate images that pass the authenticity test
    genr_loss = combined_network.train_on_batch(noise, valid_labels)
    
    # Tracking the progress
    if epoch % 100 == 0:
        print(f"[Epoch {epoch:>4}] D_loss: {discm_loss[0]:.4f} D_acc: {discm_loss[1]*100:5.2f}% G_loss: {genr_loss:.4f}")
    
    if epoch % display_interval == 0:
        display_images(generator, epoch)

# Final display and save models
print("\nTraining completed!")
display_images(generator, num_epochs)

# Save the models
generator.save(os.path.join(base_output_dir, "generator_final.h5"))
discriminator.save(os.path.join(base_output_dir, "discriminator_final.h5"))

print(f"Models and images saved to: {base_output_dir}")