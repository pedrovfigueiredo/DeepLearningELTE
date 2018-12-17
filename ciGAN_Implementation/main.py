import os, sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import initializers

from IPython.display import display
from IPython.display import Image as _Imgdis
from PIL import Image
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split

os.environ["KERAS_BACKEND"] = "tensorflow"

np.random.seed(10)

random_dim = 100

img_width = 128
img_height = 128

channels = 1

# 80% - 20%
def load_data():
    folder = "../FCN_Classifier_implementation/input/ddsm-mammography/non-negative-images"

    onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

    print("Working with {0} images".format(len(onlyfiles)))

    train_files = []
    i=0
    for _file in onlyfiles:
        train_files.append(_file)
        
    print("Files in train_files: %d" % len(train_files))

    
    nb_classes = 1

    dataset = np.ndarray(shape=(len(train_files), img_width, img_height), dtype=np.float32)

    i = 0
    for _file in train_files:
        img = load_img(folder + "/" + _file, color_mode="grayscale")  # this is a PIL image
        img.thumbnail((img_width, img_height))
        # Convert to Numpy Array
        x = img_to_array(img)
        #x = np.rollaxis(x, 2, 0)
        dataset[i] = x.reshape((img_width, img_height))
        i += 1
        if i % 250 == 0:
            print("%d images to array" % i)
    print("All images to array!")


    #Splitting 
    y_train = np.ones(dataset.shape[0])

    x_train = (dataset.astype(np.float32) - 128.0) / 128.0
    x_train = x_train.reshape(x_train.shape[0], img_height*img_width)

    return x_train, y_train


def get_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)


def get_generator(optimizer):
    generator = Sequential()
    generator.add(Dense(256, input_dim=random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(2048))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(4096))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(channels*img_width*img_height, activation='tanh'))
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return generator


def get_discriminator(optimizer):
    discriminator = Sequential()
    discriminator.add(Dense(4096, input_dim=channels*img_width*img_height, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(2048))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(1024))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return discriminator


def get_gan_network(discriminator, random_dim, generator, optimizer):
    # We initially set trainable to False since we only want to train either the
    # generator or discriminator at a time
    discriminator.trainable = False
    # gan input (noise) will be 100-dimensional vectors
    gan_input = Input(shape=(random_dim,))
    # the output of the generator (an image)
    x = generator(gan_input)
    # get the output of the discriminator (probability if the image is real or not)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    return gan


# Create a wall of generated MNIST images
def plot_generated_images(epoch, generator, examples=9, dim=(3, 3), figsize=(20, 20)):
    noise = np.random.normal(0, 1, size=[examples, random_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, img_width, img_height)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image_epoch_%d.png' % epoch)


def train(epochs=1, batch_size=128):
    # Get the training and testing data
    x_train, y_train = load_data()


    # Split the training data into batches of size 128
    batch_count = int(x_train.shape[0] / batch_size)

    # Build our GAN network
    adam = get_optimizer()
    generator = get_generator(adam)
    discriminator = get_discriminator(adam)
    gan = get_gan_network(discriminator, random_dim, generator, adam)

    for e in range(1, epochs+1):
        print('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in tqdm(range(batch_count)):
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

            # Generate fake MNIST images
            generated_images = generator.predict(noise)
            #image_batch = np.array([img.reshape((channels*img_width*img_height)) for img in image_batch])
            #generated_images = np.array([img.reshape(channels, img_width, img_height) for img in generated_images])
            X = np.concatenate([image_batch, generated_images])

            # Labels for generated and real data
            y_dis = np.zeros(2*batch_size)
            # One-sided label smoothing
            y_dis[:batch_size] = 0.9

            # Train discriminator
            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)

            # Train generator
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y_gen)

        if e == 1 or e % 50 == 0:
            plot_generated_images(e, generator)

if __name__ == '__main__':
    train(9999999, 128)