#TF Warnings Suppressor import
import TF_Warnings_Suppressor
TF_Warnings_Suppressor.configure_tensorflow()

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from datetime import datetime
import time

#Configuration Hyperparameters
buffer = 5000 #5,000 images per class in CIFAR-10
batch_size = 256
latent_noise_size = 100
epochs = 5000
examples = 16
#(TensorFlow, n.d.)

#Generated image directory path naming
script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
timestamp = datetime.now().strftime("%m-%d-%Y_%H%M")
output_dir = f'v6.2_images/{script_name}_{timestamp}'
os.makedirs(output_dir, exist_ok=True)
#(PyMOTW-3, n.d.)

#Create checkpoint directory
checkpoint_dir = './training_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)


#Load CIFAR10 data
def load_cifar10_data():
    (train_images, train_labels), (_, _) = cifar10.load_data()

    #Filter for ships (CIFAR class 8)
    ship_indices = np.where(train_labels.flatten() == 8)[0]
    train_images = train_images[ship_indices]

    #Normalize images to [-1, 1]
    train_images = train_images.reshape(train_images.shape[0], 32, 32, 3).astype('float32')
    train_images = (train_images - 127.5) / 127.5

    print(f"Loaded {len(train_images)} ship images")
    return train_images


#Generator model
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(4 * 4 * 512, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((4, 4, 512)))
    assert model.output_shape == (None, 4, 4, 512)  #Note: None is the batch size

    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 32, 32, 3)

    return model


#Discriminator model
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                            input_shape=[32, 32, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


#Loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


#Optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

#Create models
generator = make_generator_model()
discriminator = make_discriminator_model()

#Checkpoint setup
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


#Training step
@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, latent_noise_size])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


#Generate and save images
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((predictions[i] * 127.5 + 127.5).numpy().astype('uint8'))
        plt.axis('off')

    plt.savefig(f'{output_dir}/image_at_epoch_{epoch:04d}.png')
    plt.close()


#Training function
def train(dataset, epochs):
    seed = tf.random.normal([examples, latent_noise_size])

    for epoch in range(epochs):
        start = time.time()

        gen_loss_avg = 0
        disc_loss_avg = 0
        num_batches = 0

        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
            gen_loss_avg += gen_loss
            disc_loss_avg += disc_loss
            num_batches += 1

        gen_loss_avg /= num_batches
        disc_loss_avg /= num_batches

        #Generate images every 100 epochs
        if (epoch + 1) % 100 == 0:
            generate_and_save_images(generator, epoch + 1, seed)

        #Save checkpoint every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print(
            f'Epoch {epoch + 1:3d}/{epochs} - Gen Loss: {gen_loss_avg:.4f} - Disc Loss: {disc_loss_avg:.4f} - Time: {time.time() - start:.2f}s')

    #Generate final image
    generate_and_save_images(generator, epochs, seed)


def main():
    print("CIFAR10 Ships DCGAN v6.2")
    print("~~~")
    print(f"Output directory: {output_dir}")
    print("~~~")

    #Load and prepare data
    train_images = load_cifar10_data()
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer).batch(batch_size)

    print(f"Generator parameters: {generator.count_params():,}")
    print(f"Discriminator parameters: {discriminator.count_params():,}")

    #Train the model
    print(f"\nStarting training for {epochs} epochs...")
    train(train_dataset, epochs)

    print(f"\nTraining complete! Check '{output_dir}' for generated ship images.")


if __name__ == "__main__":
    main()
