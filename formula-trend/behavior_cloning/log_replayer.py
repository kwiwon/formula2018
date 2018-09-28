#!/usr/bin/env python
# -*- coding: utf-8 -*-

#######################
#
# Replay drive logs
#
#######################

import csv
from os import path

import cv2

LOG_ROOT_DIR = "<YOUR_DRIVE_LOG_FOLDER>"

csv_file_name = path.join(LOG_ROOT_DIR, "driving_log.csv")
img_dir = path.join(LOG_ROOT_DIR, "IMG")

with open(csv_file_name, newline='') as f:
    for row in csv.reader(f):
        image = cv2.imread(path.join(img_dir, row[0]))

        # Draw info
        curr_steering_angle = float(row[1])
        curr_throttle = float(row[2]) - float(row[3])
        curr_speed = float(row[4])
        cv2.putText(
            image, 'Speed: %.2f, Angle: %.2f, Throttle: %.2f' % (curr_speed, curr_steering_angle, curr_throttle),
            (10, 30), cv2.FONT_ITALIC, 0.4, (255, 0, 0), 1)

        # Draw reference line
        crop_up = 70
        crop_down = 30
        cv2.line(image, (0, crop_up), (image.shape[1], crop_up), (0, 0, 255), 1)
        cv2.line(image, (0, image.shape[0] - crop_down), (image.shape[1], image.shape[0] - crop_down), (0, 0, 255),
                 1)

        cv2.imshow('Image', image)
        cv2.waitKey(50)
