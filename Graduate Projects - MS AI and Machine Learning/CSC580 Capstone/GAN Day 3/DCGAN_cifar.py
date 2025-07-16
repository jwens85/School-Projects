#%% md
# <a href="https://colab.research.google.com/github/KeremAydin98/gan-mnist-image-generation/blob/main/DCGAN_cifar.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#%%
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
#%%
(X_train, y_train),(_,_) = cifar10.load_data()
#%%
classes={
    0 : "airplane",
    1 : "automobile",
    2 : "bird",
    3 : "cat",
    4 : "deer",
    5 : "dog",
    6 : "frog",
    7 : "horse",
    8 : "ship",
    9 : "truck"}
#%%
X_train = X_train[y_train.flatten()==7]
#%%
X_train.shape
#%%
plt.imshow(X_train[0])
#%%
X_train = X_train.astype('float32')
X_train /= 127.5
X_train -= 1
#%%
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, LeakyReLU, Reshape, BatchNormalization
from tensorflow.keras.models import Sequential
#%%
latent_dim = 32
height = 32
width = 32
channels = 3


generator = Sequential()
generator.add(Dense(128 * 16* 16, input_shape = [latent_dim]))
generator.add(LeakyReLU())
generator.add(Reshape([16,16,128]))

generator.add(Conv2D(256, 5, padding = 'same'))
generator.add(LeakyReLU())
generator.add(Conv2DTranspose(256, 4, strides = 2, padding = 'same'))
generator.add(LeakyReLU())

generator.add(Conv2D(256, 5, padding='same'))
generator.add(LeakyReLU())
generator.add(Conv2D(256, 5, padding='same'))
generator.add(LeakyReLU())

generator.add(Conv2D(channels, 7, padding='same', activation = 'tanh'))
#%%
generator.summary()
#%%
discriminator = Sequential()
discriminator.add(Conv2D(128, 3, input_shape = [width, height, channels]))
discriminator.add(LeakyReLU())

discriminator.add(Conv2D(128, 4, strides = 2))
discriminator.add(BatchNormalization())
discriminator.add(LeakyReLU())
discriminator.add(Conv2D(128, 4, strides = 2))
discriminator.add(BatchNormalization())
discriminator.add(LeakyReLU())
discriminator.add(Conv2D(128, 4, strides = 2))
discriminator.add(BatchNormalization())
discriminator.add(LeakyReLU())


discriminator.add(Flatten())
discriminator.add(Dropout(0.5))

discriminator.add(Dense(1, activation = 'sigmoid'))
#%%
discriminator.summary()
#%%
import tensorflow
discriminator_optimizer = tensorflow.keras.optimizers.RMSprop(
learning_rate=0.0002,
clipvalue=1.0,
decay=1e-8)

discriminator.compile(optimizer = discriminator_optimizer, loss = 'binary_crossentropy')

discriminator.trainable = False

GAN = Sequential([generator, discriminator])

gan_optimizer = tensorflow.keras.optimizers.RMSprop(learning_rate=0.0001, clipvalue=1.0, decay=1e-8)
GAN.compile(optimizer = gan_optimizer, loss = 'binary_crossentropy')
#%%
import numpy as np

epochs = 40000
batch_size = 20
sample_interval = 5000

generator, discriminator = GAN.layers


def train(epochs, batch_size, sample_interval):

  start = 0

  for step in range(epochs):

    for i in range(5):

      #Samples random points in the latent space
      random_latent_vectors = np.random.normal(size = (batch_size, latent_dim))

      #Decodes them to fake images
      generated_images = generator.predict(random_latent_vectors)

      #Combines them with real images
      stop = start + batch_size
      real_images = X_train[start:stop]
      combined_images = np.concatenate([generated_images, real_images])

      #Assembles labels, discriminating real images from fake images
      labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])

      #Adds random noise to labels
      labels += 0.05 * np.random.random(labels.shape)

      #Trains the discriminator
      d_loss = discriminator.train_on_batch(combined_images, labels)

    #Samples random points in the latent space(create new random points for the generator training)
    random_latent_vectors = np.random.normal(size = (batch_size, latent_dim))

    #Assembles labels that say these are all real images(it's a lie!)
    misleading_targets = np.zeros((batch_size, 1))

    #Train the generator
    g_loss = GAN.train_on_batch(random_latent_vectors, misleading_targets)

    start += batch_size
    if start > (len(X_train) - batch_size):
      start = 0

    if (step+1) % 1000 == 0:
      GAN.save_weights('cifar_gan.h5')

      print('Step: ', f"{step + 1}/{epochs}")
      print('Discriminator loss: ', d_loss)
      print('Generator loss: ', g_loss)
      print("\n")
    
    if (step+1) % sample_interval == 0:

      sample_images(generator,step+1)

#%%
from tensorflow.keras.preprocessing import image

def sample_images(generator,step):

  nb_images = 3
  noise = np.random.normal(size = (nb_images, latent_dim))
  fake_images = generator.predict(noise)

  for fake, i in zip(fake_images,range(nb_images)):
    img = image.array_to_img(fake * 255., scale=False)
    plt.imshow(img)
    plt.show()
    plt.savefig(f"{step}{i}.png")

#%%
train(epochs, batch_size, sample_interval)