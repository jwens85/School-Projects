import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, UpSampling2D, BatchNormalization, LeakyReLU

# --- 1. Data Loading and Preprocessing ---
# Load CIFAR-10 data and filter only the 'ship' class (label 8)
(X_train, y_train), (_, _) = cifar10.load_data()
ships = X_train[y_train.flatten() == 8]                     # select only class 8 (ships):contentReference[oaicite:8]{index=8}
ships = ships.astype('float32')
# Normalize images to [-1, 1] range for GAN (tanh output)
ships = (ships / 127.5) - 1.0

print(f"Training on {ships.shape[0]} ship images from CIFAR-10...")

# Image dimensions
img_rows, img_cols, channels = 32, 32, 3
image_shape = (img_rows, img_cols, channels)
latent_dim = 100  # dimension of random noise vector

# --- 2. Model Definitions ---

# Generator: converts a noise vector into a 32x32x3 image
def build_generator():
    model = Sequential(name="Generator")
    # Start with a dense layer to project the latent vector to a small spatial map
    model.add(Dense(128 * 8 * 8, activation="relu", input_dim=latent_dim))
    model.add(Reshape((8, 8, 128)))  # now shape is 8x8x128
    # Upsample to 16x16
    model.add(UpSampling2D())  # 8x8 -> 16x16
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.78))
    model.add(LeakyReLU(alpha=0.2))
    # Upsample to 32x32
    model.add(UpSampling2D())  # 16x16 -> 32x32
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.78))
    model.add(LeakyReLU(alpha=0.2))
    # Final conv to get 3-channel output
    model.add(Conv2D(3, kernel_size=3, padding="same", activation="tanh"))
    # Wrap the Sequential model with an Input to get a functional Model output
    noise = Input(shape=(latent_dim,))
    img = model(noise)
    return Model(noise, img, name="GeneratorModel")

# Discriminator: CNN that outputs probability of image being real
def build_discriminator():
    model = Sequential(name="Discriminator")
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=image_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
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
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))
    img = Input(shape=image_shape)
    validity = model(img)
    return Model(img, validity, name="DiscriminatorModel")

# Build and compile the discriminator
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                      metrics=['accuracy'])  # include accuracy for monitoring:contentReference[oaicite:9]{index=9}

# Build the generator
generator = build_generator()

# Build the combined GAN model where the generator is followed by the discriminator
# Freeze discriminator's weights in the combined model (discriminator alearning_rateeady compiled)
discriminator.trainable = False
z = Input(shape=(latent_dim,))
img = generator(z)
valid = discriminator(img)  # discriminator output on generated image
combined = Model(z, valid, name="CombinedModel")
combined.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))

# --- 3. Training Loop ---
epochs = 15000
batch_size = 32
save_interval = 2500  # interval (in iterations) to save images and metrics

# Arrays to log losses
d_losses = []
g_losses = []

# Adversarial ground truth labels (with label smoothing & noise):contentReference[oaicite:10]{index=10}
for epoch in range(epochs + 1):  # +1 so that epoch=15000 is included
    # Train Discriminator
    # Sample a random batch of real images
    idx = np.random.randint(0, ships.shape[0], batch_size)
    real_imgs = ships[idx]
    # Generate a batch of fake images
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    fake_imgs = generator.predict(noise)
    # Prepare labels for real and fake images
    real_y = np.ones((batch_size, 1)) - np.random.rand(batch_size, 1) * 0.1   # ~0.9 ± 0.1:contentReference[oaicite:11]{index=11}
    fake_y = np.random.rand(batch_size, 1) * 0.1                             # ~0.0 ± 0.1
    # Train discriminator on real and fake
    d_loss_real, d_acc_real = discriminator.train_on_batch(real_imgs, real_y)
    d_loss_fake, d_acc_fake = discriminator.train_on_batch(fake_imgs, fake_y)
    d_loss = 0.5 * (d_loss_real + d_loss_fake)
    d_acc  = 0.5 * (d_acc_real + d_acc_fake)

    # Train Generator (via combined model with frozen discriminator)
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    valid_y = np.ones((batch_size, 1))  # labels for generator training (pretend all outputs are real)
    g_loss = combined.train_on_batch(noise, valid_y)

    # Record losses for plotting
    d_losses.append(d_loss)
    g_losses.append(g_loss)

    # Print training progress occasionally
    if epoch % 500 == 0:
        print(f"Epoch {epoch:05d} / {epochs}: [D loss: {d_loss:.4f}, acc: {d_acc*100:.2f}%] [G loss: {g_loss:.4f}]")

    # Save generated images and loss plot at save_interval
    if epoch % save_interval == 0:
        # --- Save generated images grid ---
        r, c = 4, 4  # 4x4 grid
        noise = np.random.normal(0, 1, (r * c, latent_dim))
        gen_imgs = generator.predict(noise)
        # Rescale images from [-1,1] to [0,1] for viewing
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c, figsize=(4,4))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].axis('off')
                cnt += 1
        plt.tight_layout()
        plt.savefig(f"generated_epoch_{epoch:05d}.png")       # e.g., generated_epoch_00000.png:contentReference[oaicite:12]{index=12}
        plt.close()

        # --- Save training loss plot ---
        plt.figure()
        plt.plot(d_losses, label="Discriminator loss")
        plt.plot(g_losses, label="Generator loss")
        plt.title(f"GAN Loss at epoch {epoch}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"training_metrics_epoch_{epoch:05d}.png")
        plt.close()
