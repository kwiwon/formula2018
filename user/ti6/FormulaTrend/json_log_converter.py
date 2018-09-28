#!/usr/bin/env python
# -*- coding: utf-8 -*-

#####################################################################
#
# Convert TI6 drive log from JSON format to CSV with raw image files.
#
#####################################################################

import base64
import csv
import glob
import json
import math
import os
from os import path

import cv2
import numpy as np

root_dir = "/ti6_json_drive_log_folder"

csv_file_name = "driving_log.csv"
img_dir = "IMG"

json_files = glob.iglob(path.join(root_dir, '**/*.json'), recursive=True)
for json_file_path in json_files:
    root, json_log_name = path.split(json_file_path)

    print("Processing", json_file_path[len(root_dir):])

    # Create target folder
    TARGET_DIR = json_log_name.split(sep=".")[0]
    os.makedirs(path.join(root, TARGET_DIR), exist_ok=True)
    os.makedirs(path.join(root, TARGET_DIR, img_dir), exist_ok=True)

    # Parse json file
    with open(json_file_path) as f:
        json_data = json.load(f)
        lap = json_data["lap"]

        # Write CSV file
        with open(path.join(root, TARGET_DIR, csv_file_name), mode='w') as out:
            writer = csv.writer(out)

            for i, record in enumerate(json_data["records"], start=1):
                image = record["image"]
                steering_angle = record["curr_steering_angle"]
                _throttle = record["curr_throttle"]
                throttle, brake = (_throttle, 0) if _throttle > 0 else (0, math.fabs(_throttle))
                speed = record["curr_speed"]
                time = record["time"]

                # Write image to file
                img_filename = "{}_{}.jpg".format(TARGET_DIR, i)
                image_rgb = cv2.imdecode(np.fromstring(base64.b64decode(image), np.uint8), flags=cv2.IMREAD_COLOR)
                cv2.imwrite(path.join(root, TARGET_DIR, img_dir, img_filename), image_rgb,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 90])

                # Save record to CSV
                row = (img_filename,
                       steering_angle,
                       throttle,
                       brake,
                       speed,
                       time,
                       lap)
                writer.writerow(row)
