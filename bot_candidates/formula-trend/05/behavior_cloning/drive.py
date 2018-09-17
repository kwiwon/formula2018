#!/usr/bin/env python
# -*- coding: utf-8 -*-
import base64
from io import BytesIO

import numpy as np
from PIL import Image
import cv2
from keras.models import model_from_json
from .image_processor import ImageProcessor 
from behavior_cloning.train import img_pre_processing
from interface.car import Car


class BeCar(Car):
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        self.last_steering_angle = 0

    @staticmethod
    def load_model(path):
        with open(path, 'r') as jfile:
            json_model = jfile.read()
            model = model_from_json(json_model)

        model.compile("adam", "mse")
        weights_file = path.replace('json', 'h5')
        model.load_weights(weights_file)

        return model

    def print_layers(self):
        for layer in self.model.layers:
            print(layer.name, layer.input, layer.output)
            print(layer.input_shape, layer.output_shape)
            print()
            
    def make_wall_to_same_color(self, frame):
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([21, 39, 64])
        upper_yellow = np.array([40, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        frame[yellow_mask>0]=(0,0,0)
        return frame

    def wall_back(self, img, last_steering_angle):
        img.setflags(write=1)
        move_forward, px = ImageProcessor.check_recovery_direction(img)
        img = self.make_wall_to_same_color(img)
        r, g, b = cv2.split(img)
        image_height = img.shape[0]
        image_width = img.shape[1]
        image_sample = slice(0, int(image_height * 0.2))
        sr, sg, sb = r[image_sample, :], g[image_sample, :], b[image_sample, :]
        sw = sr + sg + sb
        img_area = len(sr)*len(sr[0])
        _y_w, _x_w = np.where(sw == 0)
        
        if len(_x_w) > img_area*0.8 :
            if move_forward == True:
                return last_steering_angle * 2.5
            else:
                return -last_steering_angle * 2.5
        return False


    def on_dashboard(self, data):
        # The current steering angle of the car
        # steering_angle = data["steering_angle"]
        # The current throttle of the car
        # throttle = data["throttle"]
        # The current speed of the car
        # speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)
        chk_back = self.wall_back(image_array, self.last_steering_angle)
        if chk_back != False:
            angle_ = chk_back
            throttle_ = -1
            break_ = 0
        else:
            img = img_pre_processing(image_array)
            # import time
            # from datetime import datetime
            # timestamp = int(time.mktime(datetime.now().timetuple()))
            # misc.imsave(str(timestamp)+'.png', img)

            img_batch = img[None, :, :, :].astype('float')

            prediction = self.model.predict(img_batch, batch_size=1)

            angle_ = prediction[0][0]
            throttle_ = prediction[0][1]
            break_ = prediction[0][2]
            self.last_steering_angle = angle_
            
        return angle_, (throttle_ - break_)
