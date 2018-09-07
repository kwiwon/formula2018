
import os.path
import csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout
import matplotlib.pyplot as plt


TRAIN_DIR = "C:\\TMProgram\\AIGame\\ai_global_game\\user\\mark\\formula\\data"
DRIVING_LOG = os.path.join(TRAIN_DIR, "driving_log.csv")
CORRECTION = 0.3

def extract_dl(driving_log_path):
    """Extract and load all training data in images directors and driving log csv file."""
    entries = []
    with open(driving_log_path) as csv_file:
        reader = csv.reader(csv_file)
        for entry in reader:
            entries.append(entry)
    empty_lists = [[] for i in range(7)]
    center_images, steerings, throttles, brakes, speeds, time, lap = empty_lists
    for entry in entries:
        center_image_path = entry[0]
        steering = float(entry[1])
        throttle = float(entry[2])
        brake = float(entry[3])
        speed = float(entry[4])
        center_image = cv2.imread(center_image_path)
        center_images.append(center_image)
        steerings.append(steering)
        throttles.append(throttle)
        brakes.append(brake)
        speeds.append(speed)
    return center_images, steerings, throttles, brakes, speeds

def augment_images(images, measurements, correction=0.0):
    """Augment out training image repository by adding flipped versions with inverted steering."""
    aug_imgs, aug_msrs = [], []
    for image, measurement, in zip(images, measurements):
        corr_msr = measurement + correction
        aug_imgs.append(image)
        aug_msrs.append(corr_msr)
        aug_imgs.append(cv2.flip(image, 1))
        aug_msrs.append(corr_msr*-1)
    return aug_imgs, aug_msrs


def plot_fit_history(fit_history_obj):
    """Plot loss and validation loss of the trained model on the same graph."""
    plt.plot(fit_history_obj.history['loss'])
    plt.plot(fit_history_obj.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

if __name__ == '__main__':
    cr_imgs, steerings, throttles, brakes, speeds = extract_dl(DRIVING_LOG)
    aug_cr_imgs, aug_cr_msrs = augment_images(cr_imgs, steerings)
    aug_imgs = aug_cr_imgs
    aug_msrs = aug_cr_msrs
    X_train = np.array(aug_imgs)
    y_train = np.array(aug_msrs)

    # Convolutional deep neural network based on the NVIDIA network.

    model = Sequential()
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(240, 320, 3)))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(2, 2), activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(100))
    model.add(Dropout(0.3))
    model.add(Dense(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    # Convolutional deep neural network base
    # Fit the model. No need for generator as I am using a computer with 32GB RAM
    fit_hist = model.fit(X_train, y_train, validation_split=0.2, shuffle=False, epochs=3)
    model.save('model.h5')
    model.summary()
    plot_fit_history(fit_hist)