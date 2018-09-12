#!/usr/bin/env python
# -*- coding: utf-8 -*-
import base64
from io import BytesIO

import numpy as np
from PIL import Image
from keras.models import model_from_json

from behavior_cloning.train import img_pre_processing
from interface.car import Car


class BeCar(Car):
    def __init__(self, model_path):
        self.model = self.load_model(model_path)

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

        return angle_, (throttle_ - break_)
