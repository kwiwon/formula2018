#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import getcwd, path

import eventlet
import socketio
from flask import Flask

from behavior_cloning.drive import BeCar
from pid.drive_pid import PIDCar, create_pid_driver

sio = socketio.Server()


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        steering_angle, throttle = car.on_dashboard(data)
        send_control(steering_angle, throttle)
    else:
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    print(steering_angle, throttle)
    sio.emit("steer", data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    }, skip_sid=True)


if __name__ == '__main__':
    # Create PID car
    car = PIDCar(driver=create_pid_driver())

    # Create car of behavior cloning model
    car = BeCar(model_path=path.join(getcwd(), "behavior_cloning", "model.json"))

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, Flask(__name__))

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
