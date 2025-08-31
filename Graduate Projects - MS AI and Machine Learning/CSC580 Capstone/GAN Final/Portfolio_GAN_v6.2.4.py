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
checkpoint_directory = './training_checkpoints'
os.makedirs(checkpoint_directory, exist_ok=True)
#(TensorFlow, 2024)

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
    model.add(layers.Dense(4 * 4 * 512, use_bias=False, input_shape=(latent_noise_size,)))
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
    #(TensorFlow, 2024)
    #(Radford, Metz, & Chintala, 2016)

#Discriminator model
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[32, 32, 3]))

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
    #(TensorFlow, 2024)
    #(Radford, Metz, & Chintala, 2016)

#Loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#(TensorFlow, 2024)

def discriminator_loss(true_output, synthetic_output):
    true_loss = cross_entropy(tf.ones_like(true_output), true_output)
    synthetic_loss = cross_entropy(tf.zeros_like(synthetic_output), synthetic_output)
    combined_loss = true_loss + synthetic_loss
    return combined_loss
#(TensorFlow, 2024)

def generator_loss(synthetic_output):
    return cross_entropy(tf.ones_like(synthetic_output), synthetic_output)
#(TensorFlow, 2024)

#Adam Optimizers for generator and discriminator
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
#(TensorFlow, 2024)
#(Radford, Metz, & Chintala, 2016)
#(Keras, n.d.)

#Create models
generator = make_generator_model()
discriminator = make_discriminator_model()
#(TensorFlow, 2024)

#Checkpoint setup
checkpoint_file_path = os.path.join(checkpoint_directory, "checkpoint")
model_checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
#(TensorFlow, 2024)


#Single training step function that executes forward and backward passes for the generator and discriminator
@tf.function
def perform_training_pass(images):
    noise = tf.random.normal([batch_size, latent_noise_size])

    with tf.GradientTape() as generator_gradient_tape, tf.GradientTape() as discriminator_gradient_tape:
        generated_images = generator(noise, training=True)

        true_output = discriminator(images, training=True)
        synthetic_output = discriminator(generated_images, training=True)

        generator_loss_value = generator_loss(synthetic_output)
        discriminator_loss_value = discriminator_loss(true_output, synthetic_output)

    gradients_of_generator = generator_gradient_tape.gradient(generator_loss_value, generator.trainable_variables)
    gradients_of_discriminator = discriminator_gradient_tape.gradient(discriminator_loss_value, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return generator_loss_value, discriminator_loss_value
#(TensorFlow, 2024)


#Generate and save images
def generate_and_save_images(model, epoch, test_data_input):
    generated_images = model(test_data_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(generated_images.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((generated_images[i] * 127.5 + 127.5).numpy().astype('uint8'))
        plt.axis('off')

    plt.savefig(f'{output_dir}/image_at_epoch_{epoch:04d}.png')
    plt.close()
#(TensorFlow, 2024)
#(Matplotlib, n.d.)

#Training function
def train(dataset, epochs):
    random_noise_seed = tf.random.normal([examples, latent_noise_size])

    for epoch in range(epochs):
        epoch_start_time = time.time()

        generator_loss_average = 0
        discriminator_loss_average = 0
        num_batches = 0

        for image_batch in dataset:
            generator_loss_value, discriminator_loss_value = perform_training_pass(image_batch)
            generator_loss_average += generator_loss_value
            discriminator_loss_average += discriminator_loss_value
            num_batches += 1

        generator_loss_average /= num_batches
        discriminator_loss_average /= num_batches

        #Generate images every 100 epochs and at epoch 1
        if (epoch + 1) % 100 == 0 or epoch + 1 == 1:
            generate_and_save_images(generator, epoch + 1, random_noise_seed)

        #Save checkpoint every 15 epochs
        if (epoch + 1) % 15 == 0:
            model_checkpoint.save(file_prefix=checkpoint_file_path)

        print(
            f'Epoch {epoch + 1:3d}/{epochs} - Gen Loss: {generator_loss_average:.4f}'
            f' - Disc Loss: {discriminator_loss_average:.4f} - Time: {time.time() - epoch_start_time:.2f}s')

    #Generate final image
    generate_and_save_images(generator, epochs, random_noise_seed)
#(TensorFlow, 2024)


def main():
    print("CIFAR10 Ships DCGAN v6.2.4")
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
