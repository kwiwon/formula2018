import argparse
import base64
import json
import pygame

import signal
import sys

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
from scipy import misc

from time import sleep
import threading
from keras.models import model_from_json
from train import img_pre_processing
from keras.optimizers import Adam

X_corr, y_corr = [], []
model = None
recent_angles = []
lr_val = 0
ud_val = 0
frames = 0

sio = socketio.Server(async_mode='eventlet')
app = socketio.Middleware(sio)

pygame.init()
pygame.display.set_caption('')
screen = pygame.display.set_mode((200,60), pygame.DOUBLEBUF)


def send_control(steering_angle_, throttle_, break_, nsamples=0, aaa=False):
    print('angle:', steering_angle_, 'throttle:', throttle_, 'break:', break_, 'samples:', nsamples, 'aaa:', aaa)
    sio.emit('steer', data={
        'steering_angle': steering_angle_.__str__(),
        'throttle': throttle_.__str__(),
        'break': break_.__str__(),
    }, skip_sid=True)

@sio.on('telemetry')
def telemetry(sid, data):
    global lr_val, ud_val, X_corr, y_corr, screen, frames, recent_angles
    frames += 1

    # tweak configuration
    up_limit = 0.004
    lr_limit = 15.0
    sample_interval = 10

    imgString = data['image']
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)

    img = img_pre_processing(image_array)
    img_batch = img[None, :, :, :].astype('float')

    prediction = model.predict(img_batch, batch_size=1)
    angle_ = prediction[0][0]
    throttle_ = prediction[0][1]
    break_ = prediction[0][2]

    # calculate average angle
    recent_angles.append(angle_)
    if len(recent_angles) > 10:
        recent_angles.pop(0)
    avg_angle = sum(recent_angles) / float(len(recent_angles))

    # calculate angle corrections
    key = pygame.key.get_pressed()
    if not (key[pygame.K_LEFT] or key[pygame.K_RIGHT]):
        print('lr nothing')
        lr_val = 0
    if not (key[pygame.K_UP] or key[pygame.K_DOWN]):
        print('ud nothing')
        ud_val = 0
    if key[pygame.K_LEFT]:
        print('press left')
        lr_val -= 2
        lr_val = np.max((lr_val, -lr_limit))
    if key[pygame.K_RIGHT]:
        print('press right')
        lr_val += 2
        lr_val = np.min((lr_val, lr_limit))
    if key[pygame.K_UP]:
        print('press up')
        ud_val += 0.01
        ud_val = np.min((ud_val, up_limit))
    if key[pygame.K_DOWN]:
        print('press down')
        ud_val -= 0.01
        ud_val = np.max((ud_val, -up_limit))
    if key[pygame.K_a]:
        print('apply average angle')
        lr_val = avg_angle - angle_

    # calculate corrected angle and send control
    #angle_ = np.min((np.max((angle_ + val, -0.7)), 0.7))
    angle_ = np.min((np.max((angle_ + lr_val, -45.0)), 45.0))
    throttle_ = np.min((np.max((throttle_ + ud_val, -10.0)), 10.0))
    
    # save corrections
    if lr_val != 0 or ud_val != 0:
        print('corrections')
        X_corr.append(img.astype('float'))
        y_corr.append([angle_, throttle_, break_])
    # sample old model
    elif frames % sample_interval == 0 or key[pygame.K_SPACE]:
        print('sample old model')
        X_corr.append(img.astype('float'))
        y_corr.append([angle_, throttle_, break_])

    # control car
    send_control(angle_, throttle_, break_, len(X_corr), True)

    surface = pygame.surfarray.make_surface(np.flipud(np.rot90(img)))
    screen.blit(surface, (0, 0))
    pygame.display.flip()

    # refresh keyboard state
    pygame.event.pump()


def signal_handler(signal, frame):
    global X_corr, y_corr, screen
    print('Found', len(X_corr), 'to train with')

    if len(X_corr) > 0:
        X_corr = np.array(X_corr, dtype='float')
        y_corr = np.array(y_corr, dtype='float')
        print('Training with', X_corr.shape[0], 'samples')

        # for VGG we choose to include the
        # top 2 blocks in training
        for layer in model.layers[:11]:
           layer.trainable = False
        for layer in model.layers[11:]:
           layer.trainable = True

        model.fit(X_corr, y_corr,
                  batch_size=128,
                  nb_epoch=10,
                  verbose=1)

        timestamp = str(int(time.time()))
        json_filename = 'model_' + timestamp + '.json'
        weights_filename = 'model_' + timestamp + '.h5'

        model_json = model.to_json()
        with open(json_filename, 'w') as json_file:
            json_file.write(model_json)
        model.save_weights(weights_filename)

        print('Model saved at: ', json_filename)
        print('Weights saved at: ', weights_filename)

    sys.exit(0)


@sio.on('connect')
def connect(sid, environ):
    print('connect ', sid)
    send_control(0, 0, 0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        json_model = jfile.read()
        model = model_from_json(json_model)

    opt = Adam(lr=1e-05, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    model.compile(optimizer=opt, loss='mse')

    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # deploy as an eventlet WSGI server
    signal.signal(signal.SIGINT, signal_handler)
    print('''
    Usage : click pygame window and press keys to tweak model
          <Arrow key> : Tweak throttle and angle
          <Space>     : Record old model behavior as training data
          <a>         : Apply average angle of recent 10 frames
          <Ctrl+C>    : Start to train
          ''')
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
