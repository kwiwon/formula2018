import tensorflow as tf
import pandas as pd
import numpy as np
import os 
import glob
import json 

from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dropout, Activation, Lambda
from keras.layers import Input, Flatten, Dense, ELU
from keras.callbacks import EarlyStopping

from scipy import misc
from PIL import Image 
from io import BytesIO 
import cv2
from augmentation import augment_image

import base64
flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_integer('features_epochs', 1,
                     'The number of epochs when training features.')
flags.DEFINE_integer('full_epochs', 10000,
                     'The number of epochs when end-to-end training.')
flags.DEFINE_integer('batch_size', 64, 'The batch size.')
flags.DEFINE_integer('samples_per_epoch', 2560,
                     'The number of samples per epoch.')
flags.DEFINE_integer('img_h', 70, 'The image height.')
#flags.DEFINE_integer('img_h', 100, 'The image height.')
flags.DEFINE_integer('img_w', 200, 'The image width.')
flags.DEFINE_integer('img_c', 3, 'The number of channels.')

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()    
    keys_list = [keys for keys in flags_dict]    
    for keys in keys_list:    
        FLAGS.__delattr__(keys)


def img_pre_processing(img, old = False):

    if old:
        # resize and cast to float
        img = misc.imresize(
            img, (150, FLAGS.img_w)).astype('float')
    else:
        # resize and cast to float
        img = misc.imresize(
            img, (110, FLAGS.img_w)).astype('float')
        img = img[40:]

    # normalize
    img /= 255.
    img -= 0.5
    img *= 2.
    return img

def img_paths_to_img_array(image_paths):
    all_imgs = [misc.imread(imp) for imp in image_paths]
    return np.array(all_imgs, dtype='float')

def save_model(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights('model.h5')

def select_specific_set(iter_set):
    imgs, labs = [], []
    for _, row in iter_set:
        # extract the features and labels
        #  img_f = 'data' + row['img'].split('../..')[1]
        #  img_ = img_pre_processing(misc.imread(img_f))
        img_= row['img']
        img_ = img_pre_processing(img_)
        angle_ = row['angle']
        throttle_ = row['throttle']
        break_ = row['break']

        # flip 50% of the time
        if np.random.choice([True, False]):
            img_, angle_ = np.fliplr(img_), -angle_ + 0.

            
        imgs.append(img_)
        labs.append([angle_, throttle_, break_])

    return np.array(imgs), np.array(labs)

def generate_batch(log_data):
    while True:
        imgs, labs = select_specific_set(
            log_data.sample(
                32).iterrows())
        yield np.array(imgs), np.array(labs)

def color_percentage(img):
    LowerYellow = np.array([26, 43, 46])
    UpperYellow = np.array([34, 255, 255])
    img_wall_yellow = cv2.inRange(img, LowerYellow, UpperYellow)
    
    ratio =cv2.countNonZero(img_wall_yellow)/(img.size/3)
    #  print("yellow ratio percentage: ", np.round(ratio*100, 2))
    return np.round(ratio*100, 2)

def main(_):
    #  del_all_flags(tf.flags.FLAGS)
    import glob
    # fix random seed for reproducibility
    np.random.seed(123)

    record_folder = ["./drive/data/driving-records-20180906/Track5",                          
                 "drive/data/driving-records-20180906/Track4",
                 "drive/data/driving-records-20180906/Track3",
                #  "drive/data/driving-records-20180906/Track2",
                 "drive/data/driving-records-20180906/Track1"]
    record_file_to_play = []
    lab_data = []
    for folder in record_folder:
        files = glob.glob(os.path.join(folder, '*flip.json'))
        files = sorted(files)
        record_file_to_play.extend(files)

    for record_file in record_file_to_play:
        last_speed = 0
        with open(record_file) as f:
            record_json = json.load(f)
        
        for record in record_json['records']:
            row = {}
            img = np.asarray(Image.open(
                BytesIO(base64.b64decode(record["image"]))))
            image_rgb = cv2.imdecode(np.fromstring(base64.b64decode(record["image"]), np.uint8), flags=cv2.IMREAD_COLOR)
            color_percentage_y = color_percentage(image_rgb)
    
            steering_angle = record['curr_steering_angle']
            throttle = record['curr_throttle']
            current_speed = record['curr_speed']
            blake =  0 if current_speed > last_speed else 1
            row['img'], row['angle'], row['throttle'] = augment_image(img, steering_angle, throttle)
            row['break'] = blake 
            row['speed'] = current_speed 
            row['time'] = record['time'] 
            row['lap'] = record_json['lap']
    
            # 轉彎 與 小路判斷, 加重資料的比例
            if np.abs(steering_angle) > 10 or color_percentage_y > 0.01:
                cnt = 10
                while cnt > 0:
                    lab_data.append(row)
                    cnt -= 1

    print("Got", len(lab_data), "samples for training")
    log_data = pd.DataFrame(lab_data)         
    X_val, y_val = select_specific_set(
        log_data.sample(int(len(log_data)*.10)).iterrows())
    print("Got", len(X_val), "samples for validation")

    # create and train the model
    input_shape = (FLAGS.img_h, FLAGS.img_w, FLAGS.img_c)
    input_tensor = Input(shape=input_shape)

    # get the VGG16 network
    base_model = VGG16(input_tensor=input_tensor,
                             weights='imagenet',
                             include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)


    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    predictions = Dense(3, init='zero')(x)

    model = Model(input=base_model.input, output=predictions)

    # freeze all convolutional layers to initialize the top layers
    for layer in base_model.layers:
        layer.trainable = False

    # train the top layer to prepare all weights
    model.compile(optimizer='adam', loss='mse')

    print('Train fully-connected layers weights:')
    history = model.fit_generator(
        generate_batch(log_data),
        samples_per_epoch=10000,
        nb_epoch=FLAGS.features_epochs,
        verbose=1)

    # print all layers
    print("Network architecture:")
    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    # for VGG we choose to include the
    # top 2 blocks in training
    for layer in model.layers[:11]:
       layer.trainable = False
    for layer in model.layers[11:]:
       layer.trainable = True

    # recompile and train with a finer learning rate
    opt = Adam(lr=1e-03, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.7)
    model.compile(optimizer=opt, loss='mse')
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=1,
                                   min_delta=0.00009)

    print('Train top 2 conv blocks and fully-connected layers:')
    history = model.fit_generator(
        generate_batch(log_data),
        samples_per_epoch=FLAGS.samples_per_epoch,
        validation_data=(X_val, y_val),
        nb_epoch=FLAGS.full_epochs,
        callbacks=[early_stopping],
        verbose=1)

    # save model to disk
    save_model(model)
    print('model saved')


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()

