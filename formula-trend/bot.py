#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import getcwd, path

import eventlet
import socketio
from flask import Flask
import sys, getopt

sio = socketio.Server()
car = None
app = None

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
    # print(steering_angle, throttle)
    sio.emit("steer", data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    }, skip_sid=True)


def main(argv):
    global car
    global app
    try:
        opts, args = getopt.getopt(argv, "hb:")
    except getopt.GetoptError:
        print('bot.py -b <bot_type>')
        sys.exit(2)
    bot_type = None
    for opt, arg in opts:
        if opt == '-h':
            print('bot.py -b <bot_type>')
            print('bot_type: bc, mpc, pid')
            sys.exit()
        elif opt == "-b":
            bot_type = arg

    if bot_type == 'bc':
        from behavior_cloning.drive import BeCar
        # Create car of behavior cloning model
        car = BeCar(model_path=path.join(getcwd(), "behavior_cloning", "model.json"))
    elif bot_type == 'mpc':
        from mpc.drive_mpc import MpcCar, create_mpc_driver
        # Create MPC car
        car = MpcCar(driver=create_mpc_driver(lib_dir=path.join(getcwd(), "mpc")))
    elif bot_type == 'pid':
        from pid.drive_pid import PIDCar, create_pid_driver
        # Create PID car
        car = PIDCar(driver=create_pid_driver(do_sign_detection=False))
    else:
        print('bot type should be in (bc, mpc, pid)')
        sys.exit()

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, Flask(__name__))

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)


if __name__ == "__main__":
    argv = sys.argv[1:]
    sys.argv = sys.argv[:1]
    main(argv)
