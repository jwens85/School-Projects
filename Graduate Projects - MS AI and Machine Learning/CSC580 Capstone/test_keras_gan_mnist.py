import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Flatten, LeakyReLU
from keras.optimizers import Adam

# Test parameters
latent_dim = 100
epochs = 500  # Short test
batch_size = 32

# Output directory
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
output_dir = os.path.join("outputs", "keras_test", timestamp)
os.makedirs(output_dir, exist_ok=True)

# Load MNIST data
(X_train, _), (_, _) = mnist.load_data()
X_train = X_train.astype('float32') / 255.0
X_train = (X_train - 0.5) * 2  # Normalize to [-1, 1]
X_train = X_train.reshape(X_train.shape[0], 784)

print(f"Loaded {len(X_train)} MNIST images")

# Simple generator
def build_generator():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=latent_dim))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(784, activation='tanh'))
    return model

# Simple discriminator  
def build_discriminator():
    model = Sequential()
    model.add(Dense(256, activation='relu', input_dim=784))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Build models
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002), metrics=['accuracy'])

generator = build_generator()

# Combined model
z = Input(shape=(latent_dim,))
img = generator(z)
discriminator.trainable = False
validity = discriminator(img)
combined = Model(z, validity)
combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002))

print("Starting MNIST GAN test...")

# Training loop
for epoch in range(epochs):
    # Train discriminator
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_imgs = X_train[idx]
    
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    fake_imgs = generator.predict(noise, verbose=0)
    
    real_labels = np.ones((batch_size, 1)) * 0.9
    fake_labels = np.zeros((batch_size, 1))
    
    d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
    # Train generator
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    valid_y = np.ones((batch_size, 1))
    g_loss = combined.train_on_batch(noise, valid_y)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: D_loss: {d_loss[0]:.4f}, D_acc: {d_loss[1]*100:.1f}%, G_loss: {g_loss:.4f}")
        
        # Save sample images
        noise = np.random.normal(0, 1, (16, latent_dim))
        gen_imgs = generator.predict(noise, verbose=0)
        gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale to [0, 1]
        
        fig, axs = plt.subplots(4, 4, figsize=(8, 8))
        count = 0
        for i in range(4):
            for j in range(4):
                axs[i,j].imshow(gen_imgs[count].reshape(28, 28), cmap='gray')
                axs[i,j].axis('off')
                count += 1
        
        plt.suptitle(f'MNIST Test - Epoch {epoch}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"epoch_{epoch}.png"))
        plt.close()

print(f"MNIST GAN test completed! Check results in: {output_dir}")
print("If you see recognizable digits, Keras GANs work in your environment!")
print("If you see mode collapse/noise, there's a fundamental Keras issue.")