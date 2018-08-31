# coding: utf-8

import os
import random
from math import e, sqrt, pi

import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Cropping2D
from keras.layers import Dense
from keras.layers import Lambda
from keras.layers import concatenate
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.pooling import AveragePooling2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.utils import Sequence
from keras.utils.vis_utils import plot_model as plot
from scipy.misc import imread
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Fix random seed for reproducibility
seed = 7
random.seed(seed)
np.random.seed(seed)
random_state = 42

# Look for GPU
# device_name = tf.test.gpu_device_name()
# if device_name != '/device:GPU:0':
#     raise SystemError('GPU device not found')
# print('Found GPU at: {}'.format(device_name))

# Define data folder
folder = '/content/data'
img_path = folder + '/IMG/'
driving_log = folder + '/driving_log.csv'


########################
# Utility functions
########################
def gauss(x, mu=0, sigma=0.18):
    """
    utility function to calculate gaussion function
    """
    a = 1 / (sigma * sqrt(2 * pi))
    return a * e ** (-0.5 * (float(x - mu) / sigma) ** 2)


max_gauss = gauss(0)


def _should_drop(steering, drop_rate=0.7):
    """
    Randomly drop some data that drives around 0 degree
    (in a normal distribution manager.)

    for more detail please see writeup_report.md
    """
    steer_drop_rate = drop_rate * gauss(steering) / max_gauss
    return random.random() < steer_drop_rate


def load_data(file_path):
    print('Reading data from %s ...' % file_path)

    data = []

    def get_img_file_name(path):
        return os.path.split(path)[-1]

    with open(file_path, 'r') as f:
        # there is no header name
        for row in f:
            (center, steering, throttle,
             brake, speed, time, lap) = row.strip().split(",")

            # randomly skip some data driving strait
            steering = float(steering)
            if _should_drop(steering):
                continue
            else:
                data.append((get_img_file_name(center), steering, throttle,
                             brake, speed, time, lap))

    # Convert list to 2-D nsarray
    return np.asarray(data)


def load_img_data(img_name, steering, random_flip=False, img_path=img_path):
    """Load image data (and randomly flip if required)"""
    img = imread(img_path + img_name).astype(np.float32)

    if random_flip and random.random() > 0.5:
        img = np.fliplr(img)
        steering = -steering

    return img, steering


def _normalize(X):
    a = -1.0
    b = 1.0
    x_min = 0
    x_max = 255
    return a + (X - x_min) * (b - a) / (x_max - x_min)


########################
# Model / Data Generator
########################
def model_builder():
    # if os.path.exists(folder + '/model.h5'):
    #     return load_model(folder + '/model.h5')

    """
    Define and compile model
    """
    main_input = Input(shape=(240, 320, 3), name='main_input')

    # crop image 3@160x320 -> 3@80x320
    x = Cropping2D(
        cropping=((50, 30), (0, 0)),
        input_shape=(240, 320, 3))(main_input)
    # normalize rgb data [0~255] to [-1~1]
    x = Lambda(_normalize, name='RGB_Normalize')(x)

    # Convolution layers
    # Let network learn it's own color spaces
    x = Convolution2D(3, (1, 1))(x)
    # reshape image by 1/4 using average pooling later
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    # 3@40x160
    x = Convolution2D(24, (5, 5), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    # 24@18x78
    x = Convolution2D(36, (5, 5), activation='relu')(x)
    x = MaxPooling2D((2, 2), (1, 2))(x)
    # 36@7x37
    x = Convolution2D(48, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    # 48@5x35
    x = Convolution2D(64, (3, 3), activation='relu')(x)
    # 64@3x33
    x = Convolution2D(64, (3, 3), activation='relu')(x)
    # 64@1x31

    # Fully connected layers
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(100, activation='relu')(x)
    x_out = Dropout(0.5)(x)

    # Another input for driving info
    driving_input = Input(shape=(1,), name='driving_input')

    # We stack a deep densely-connected network on top
    x = concatenate([x_out, driving_input])
    x = Dense(50, activation='relu')(x)
    x = Dense(10, activation='relu')(x)

    main_output = Dense(2, activation='linear', name='main_output')(x)

    model = Model(inputs=[main_input, driving_input], outputs=[main_output])

    adam = Adam(decay=0.0)
    model.compile(adam, loss='mse', metrics=['mse'])
    return model


class SequenceData(Sequence):
    def __init__(self, data, batch_size=128, training=False):
        self.data = data
        self.batch_size = batch_size
        self.training = training

    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def __getitem__(self, idx):
        X_main_batch = []     # main input
        X_driving_batch = []  # driving input
        y_batch = []          # main output (labels)

        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        batch_data = self.data[start:end]

        for item in batch_data:
            # item: center, steering, throttle, brake, speed, time, lap
            # Remember to cast data to numeric value
            steering = float(item[1])
            throttle = float(item[2])
            brake = float(item[3])
            speed = float(item[4])
            img, steering = load_img_data(item[0], steering=steering, random_flip=self.training)

            # Input/Output for model
            X_main_batch.append(img)
            X_driving_batch.append(speed)
            y_batch.append((steering, throttle - brake))

        return [np.array(X_main_batch), np.asarray(X_driving_batch)], np.array(y_batch)


if __name__ == '__main__':
    # Preprocess:
    # read driving_log.csv and prepare training dataset
    dataset = load_data(driving_log)

    # Split into train and test data (2-D array)
    print('Shuffling and Train test split ...')
    dataset = shuffle(dataset, random_state=random_state)
    data_train, data_test = train_test_split(dataset, test_size=0.2, random_state=random_state)

    print('Validating traing / testing data size ...')
    print('training set size %d' % len(data_train))
    print('testing set size %d' % len(data_test))
    print('Data looks good!')

    # Create model
    print('Creating model...')
    model = model_builder()
    # plot(model, to_file='model.png', show_shapes=True, show_layer_names=False)

    # Train model
    batch_size = 32
    init_epoch = 0
    nb_epochs = 1

    print('Start training... batch size %d' % batch_size)
    train_generator = SequenceData(data_train, batch_size=batch_size, training=True)
    test_generator = SequenceData(data_test, batch_size=batch_size)
    save_checkpoint = ModelCheckpoint(folder + '/checkpoint.{epoch:02d}.h5', period=100)

    print("Model fitting...")
    model.fit_generator(
        train_generator, epochs=nb_epochs, initial_epoch=init_epoch,
        validation_data=test_generator,
        callbacks=[save_checkpoint])
    print('Finished!')

    # Save trained model
    model_save_name = folder + '/model.h5'
    print('Saving model...')
    model.save(model_save_name)
    print('Model has been save as %s' % model_save_name)
