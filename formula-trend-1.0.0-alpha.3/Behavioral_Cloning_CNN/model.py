# coding: utf-8

from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers.pooling import AveragePooling2D
from keras.layers import Cropping2D
from keras.layers import Lambda
from keras.callbacks import ModelCheckpoint
from tensorflow.python.client import device_lib
from scipy.misc import imread
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.utils.vis_utils import plot_model as plot
from math import e, sqrt, pi

import random
import numpy as np
import csv
import os
import tensorflow as tf
import re


def has_gpu():
    local_device_protos = device_lib.list_local_devices()
    return True if [x.name for x in local_device_protos if x.device_type == 'GPU'] != [] else False

# If has gpu, control GPU Memory
if has_gpu:
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    tf.keras.backend.set_session(sess)





img_path = './data/img/'
driving_log = './data/driving_log.csv'

def get_img_file_name(path):
    return os.path.split(path)[-1]

def gauss(x, mu=0, sigma=0.18):
    """
    utility function to calculate gaussion function
    """
    a = 1/(sigma*sqrt(2*pi))
    return a*e**(-0.5*(float(x-mu)/sigma)**2)

random_state = 42
current_dir = os.getcwd()
max_gauss = gauss(0)

images = []
steerings = []


def _should_drop(steering, drop_rate=0.7):
    """
    Randomly drop some data that drives around 0 degree
    (in a normal distribution manager.)

    for more detail please see writeup_report.md
    """
    steer_drop_rate = drop_rate * gauss(steering) / max_gauss
    return random.random() < steer_drop_rate

# Preprocess:
# read driving_log.csv and prepare training dataset
print('Reading data from %s ...' % driving_log)

with open(driving_log, 'r') as f:
    # there is no header name
    for row in f.readlines():
        (center, steering, throttle,
            brake, speed, time, lap) = row.split(",")
        steering = float(steering)
        center = get_img_file_name(center)
       
        # randomly skip some data driving strait
        if _should_drop(steering):
            continue
        else:
            images.append(center)
            steerings.append(steering)





def prepare_data(img_name, steering, random_flip=False, img_path=img_path):
    """Load image data (and randomly flip if required)"""
    img = imread(img_path+img_name).astype(np.float32)

    if random_flip and random.random() > 0.5:
        img = np.fliplr(img)
        steering = -steering

    return img, steering


# generator function for training and validating
def batches(img_names, steerings, batch_size=128, training=False):
    """Generator that generates data batch by batch
    validating: indicates generator is in training mode
    """
    # check input data integrity
    num_imgs = img_names.shape[0]
    num_steerings = steerings.shape[0]
    assert num_imgs == num_steerings

    while True:
        for offset in range(0, num_imgs, batch_size):
            X_batch = []
            y_batch = []

            stop = offset + batch_size
            img_names_b = img_names[offset:stop]
            steerings_b = steerings[offset:stop]

            for i in range(img_names_b.shape[0]):
                img, steering = prepare_data(
                    img_names_b[i], steerings_b[i], random_flip=training)
                X_batch.append(img)
                y_batch.append(steering)

            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)
            yield X_batch, y_batch


def _normalize(X):
    a = -0.1
    b = 0.1
    x_min = 0
    x_max = 255
    return a + (X - x_min) * (b - a) / (x_max - x_min)


def model_builder():
    """
    Define and compile model
    """
    model = Sequential()
    # crop image 3@160x320 -> 3@80x320
    model.add(Cropping2D(
        cropping=((50, 30), (0, 0)),
        input_shape=(240, 320, 3)))
    # normalize rgb data [0~255] to [-1~1]
    model.add(Lambda(_normalize))

    # Convolution layers
    # Let network learn it's own color spaces
    model.add(Convolution2D(3, (1, 1)))
    # reshape image by 1/4 using average pooling later
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    # 3@40x160
    model.add(Convolution2D(24, (5, 5), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    # 24@18x78
    model.add(Convolution2D(36, (5, 5), activation='relu'))
    model.add(MaxPooling2D((2, 2), (1, 2)))
    # 36@7x37
    model.add(Convolution2D(48, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    # 48@5x35
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    # 64@3x33
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    # 64@1x31

    # Fully connected layers
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    model.compile('Adam', 'mse', metrics=['mse'])
    return model 





print('Shuffling and Train test split ...')
images, steerings = shuffle(images, steerings, random_state=random_state)
paths_train, paths_test, steerings_train, steerings_test = train_test_split(
    images, steerings, test_size=0.2, random_state=random_state)





# check testing data ok
paths_test = np.array(paths_test)
steerings_test = np.array(steerings_test)
assert paths_test.shape[0] == steerings_test.shape[0]
print('validation set size %d' % steerings_test.shape[0])

# check training data ok
paths_train = np.array(paths_train)
steerings_train = np.array(steerings_train)
assert paths_train.shape[0] == steerings_train.shape[0]
print('training set size %d' % paths_train.shape[0])





print('Creating model...')

model = model_builder()
plot(model, to_file='model.png', show_shapes=True, show_layer_names=False)

# Train model
print('Validating traing / testing data size ...')
assert paths_train.shape[0] == steerings_train.shape[0]
assert paths_test.shape[0] == steerings_test.shape[0]
print('Data looks good!')

train_size = paths_train.shape[0]
test_size = paths_test.shape[0]
batch_size = 128
nb_epochs = 10

print('Start training... batch size %d' % batch_size)
train_generator = batches(
    paths_train, steerings_train, batch_size=batch_size, training=True)
test_generator = batches(paths_test, steerings_test, batch_size=batch_size)

save_checkpoint = ModelCheckpoint('checkpoint.{epoch:02d}.h5', period=5)

print("Model fitting...")
model.fit_generator(
    train_generator, train_size, nb_epochs,
    validation_data=test_generator,
    nb_val_samples=test_size,
    callbacks=[save_checkpoint])
print('Finished!')





# Save trained model
model_save_name = 'model.h5'
print('Saving model...')
model.save(model_save_name)
print('Model has been save as %s', model_save_name)

