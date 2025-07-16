from discriminator import build_discriminator
from generator import build_generator


from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input

latent_dimensions = 100

# Compile discriminator first
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(0.0002, 0.5),
                      metrics=['accuracy'])

# Build generator
generator = build_generator()

# Freeze discriminator weights for generator training
discriminator.trainable = False

# Create GAN pipeline: input is noise, output is discriminator's classification of generated image
z = Input(shape=(latent_dimensions,))
img = generator(z)
validity = discriminator(img)

# Combined model (generator + discriminator)
combined_network = Model(z, validity)
combined_network.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
