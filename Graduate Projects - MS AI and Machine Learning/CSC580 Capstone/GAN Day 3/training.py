import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam

from generator import build_generator
from discriminator import build_discriminator

latent_dimensions = 100

# --- Build Models ---
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(0.0002, 0.5),
                      metrics=['accuracy'])

generator = build_generator()

# Freeze discriminator during generator training
discriminator.trainable = False

z = Input(shape=(latent_dimensions,))
img = generator(z)
validity = discriminator(img)

combined_network = Model(z, validity)
combined_network.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# --- Load and Filter CIFAR-10 (class 8: horses) ---
(X_train, y_train), (_, _) = cifar10.load_data()
X_train = X_train[y_train.flatten() == 8]
X_train = (X_train / 127.5) - 1.0  # Normalize to [-1, 1]

# --- Training Parameters ---
num_epochs = 15000
batch_size = 32
display_interval = 2500

# Preallocate noisy labels
valid = np.ones((batch_size, 1)) + 0.05 * np.random.random((batch_size, 1))
fake = np.zeros((batch_size, 1)) + 0.05 * np.random.random((batch_size, 1))

# --- Training Loop ---
for epoch in range(num_epochs):

    # --- Train Discriminator ---
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_imgs = X_train[idx]

    noise = np.random.normal(0, 1, (batch_size, latent_dimensions))
    gen_imgs = generator.predict(noise, verbose=0)

    d_loss_real = discriminator.train_on_batch(real_imgs, valid)
    d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # --- Train Generator ---
    noise = np.random.normal(0, 1, (batch_size, latent_dimensions))
    g_loss = combined_network.train_on_batch(noise, valid)

    # --- Logging ---
    if epoch % 100 == 0:
        print(f"{epoch} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

    # --- Display Generated Images ---
    if epoch % display_interval == 0:
        r, c = 4, 4
        noise = np.random.normal(0, 1, (r * c, latent_dimensions))
        gen_imgs = generator.predict(noise, verbose=0)
        gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale to [0, 1]

        fig, axs = plt.subplots(r, c)
        count = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[count])
                axs[i, j].axis("off")
                count += 1
        plt.show()
        plt.close()
