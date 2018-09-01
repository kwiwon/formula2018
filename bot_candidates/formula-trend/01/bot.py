import base64
from io import BytesIO

import eventlet.wsgi
import numpy as np
import socketio
from PIL import Image
from flask import Flask
from keras.models import load_model

# Fix error with Keras and TensorFlow
# import tensorflow as tf
# tf.python.control_flow_ops = tf

sio = socketio.Server()


@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = float(data["speed"])
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    transformed_image_array = image_array[None, :, :, :]

    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    predictions = model.predict(transformed_image_array, batch_size=1)
    steering_angle = float(predictions[0][0])

    # Use rule-based throttle per steering_angle
    if abs(steering_angle) > 5.0:
        throttle = 0.001
    else:
        throttle = max(0.01, -0.15 / 0.05 * abs(steering_angle) + 0.35)

    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    print(steering_angle, throttle)
    sio.emit("steer", data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    model = load_model('./model.h5')

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, Flask(__name__))

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
