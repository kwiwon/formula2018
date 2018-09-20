#!env python
#
# Auto-driving Bot
#
# Revision:      v1.2
# Released Date: Aug 20, 2018
#
from __future__ import print_function

import argparse
import base64
import copy
import json
import math
import os
# import datetime
# import matplotlib.pyplot as plt
from ctypes import *
from io import BytesIO
from sys import platform
from time import time

import cv2
import eventlet.wsgi
import numpy as np
import socketio
from PIL import Image
from enum import Enum
from flask import Flask

from interface.car import Car


class MPC_MODE(Enum):
    mpc_control_by_line = 0
    mpc_control_by_color = 1


USE_MPC = True
REF_LINE = None


def logit(msg):
    print("%s" % msg)


class MPC(object):
    def __init__(self, lib_path, model_settings_path, debug=False):
        self.mpc_lib = cdll.LoadLibrary(lib_path)
        self.mpc_lib.ChangeSettings(c_char_p(model_settings_path))
        self.debug = debug

    def run(self, ptsx, ptsy, v):
        telemetry = {"ptsx": ptsx, "ptsy": ptsy, "speed": v}
        if self.debug:
            print(telemetry)
        # self.mpc_lib.Predict.argtypes = [c_char_p]
        self.mpc_lib.Predict.restype = c_char_p
        res = self.mpc_lib.Predict(c_char_p(json.dumps(telemetry)))
        if self.debug:
            print(res)
        return res


class PID:
    def __init__(self, Kp, Ki, Kd, max_integral, min_interval=0.001, set_point=0.0, last_time=None):
        self._Kp = Kp
        self._Ki = Ki
        self._Kd = Kd
        self._min_interval = min_interval
        self._max_integral = max_integral

        self._set_point = set_point
        self._last_time = last_time if last_time is not None else time()
        self._p_value = 0.0
        self._i_value = 0.0
        self._d_value = 0.0
        self._d_time = 0.0
        self._d_error = 0.0
        self._last_error = 0.0
        self._output = 0.0

    def update(self, cur_value, cur_time=None):
        if cur_time is None:
            cur_time = time()

        error = self._set_point - cur_value
        d_time = cur_time - self._last_time
        d_error = error - self._last_error

        if d_time >= self._min_interval:
            self._p_value = error
            self._i_value = min(max(error * d_time, -self._max_integral), self._max_integral)
            self._d_value = d_error / d_time if d_time > 0 else 0.0
            self._output = self._p_value * self._Kp + self._i_value * self._Ki + self._d_value * self._Kd

            self._d_time = d_time
            self._d_error = d_error
            self._last_time = cur_time
            self._last_error = error

        return self._output

    def reset(self, last_time=None, set_point=0.0):
        self._set_point = set_point
        self._last_time = last_time if last_time is not None else time()
        self._p_value = 0.0
        self._i_value = 0.0
        self._d_value = 0.0
        self._d_time = 0.0
        self._d_error = 0.0
        self._last_error = 0.0
        self._output = 0.0

    def assign_set_point(self, set_point):
        self._set_point = set_point

    def get_set_point(self):
        return self._set_point

    def get_p_value(self):
        return self._p_value

    def get_i_value(self):
        return self._i_value

    def get_d_value(self):
        return self._d_value

    def get_delta_time(self):
        return self._d_time

    def get_delta_error(self):
        return self._d_error

    def get_last_error(self):
        return self._last_error

    def get_last_time(self):
        return self._last_time

    def get_output(self):
        return self._output


class ImageProcessor(object):
    @staticmethod
    def show_image(img, name="image", scale=1.0):
        if scale and scale != 1.0:
            img = cv2.resize(img, newsize, interpolation=cv2.INTER_CUBIC)

        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(name, img)
        cv2.waitKey(1)

    @staticmethod
    def save_image(folder, img, prefix="img", suffix=""):
        from datetime import datetime
        filename = "%s-%s%s.jpg" % (prefix, datetime.now().strftime('%Y%m%d-%H%M%S-%f'), suffix)
        cv2.imwrite(os.path.join(folder, filename), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    @staticmethod
    def rad2deg(radius):
        return radius / np.pi * 180.0

    @staticmethod
    def deg2rad(degree):
        return degree / 180.0 * np.pi

    @staticmethod
    def bgr2rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _normalize_brightness(img):
        maximum = img.max()
        if maximum == 0:
            return img
        adjustment = min(255.0 / img.max(), 3.0)
        normalized = np.clip(img * adjustment, 0, 255)
        normalized = np.array(normalized, dtype=np.uint8)
        return normalized

    @staticmethod
    def _flatten_rgb(img):
        r, g, b = cv2.split(img)
        r_filter = (r == np.maximum(np.maximum(r, g), b)) & (r >= 120) & (g < 150) & (b < 150)
        g_filter = (g == np.maximum(np.maximum(r, g), b)) & (g >= 120) & (r < 150) & (b < 150)
        b_filter = (b == np.maximum(np.maximum(r, g), b)) & (b >= 120) & (r < 150) & (g < 150)
        y_filter = ((r >= 128) & (g >= 128) & (b < 100))

        r[y_filter], g[y_filter] = 255, 255
        b[np.invert(y_filter)] = 0

        b[b_filter], b[np.invert(b_filter)] = 255, 0
        r[r_filter], r[np.invert(r_filter)] = 255, 0
        g[g_filter], g[np.invert(g_filter)] = 255, 0

        flattened = cv2.merge((r, g, b))
        return flattened

    @staticmethod
    def _crop_image(img,ratio=0.55):
        bottom_half_ratios = (ratio, 1.0)
        bottom_half_slice = slice(*(int(x * img.shape[0]) for x in bottom_half_ratios))
        bottom_half = img[bottom_half_slice, :, :]
        return bottom_half

    @staticmethod
    def preprocess(img, ratio=0.55):
        img = ImageProcessor._crop_image(img, ratio)
        img = ImageProcessor._normalize_brightness(img)
        img = ImageProcessor._flatten_rgb(img)
        return img

    @staticmethod
    def find_lines(img):
        grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grayed, (3, 3), 0)
        # edged      = cv2.Canny(blurred, 0, 150)

        sobel_x = cv2.Sobel(blurred, cv2.CV_16S, 1, 0)
        sobel_y = cv2.Sobel(blurred, cv2.CV_16S, 0, 1)
        sobel_abs_x = cv2.convertScaleAbs(sobel_x)
        sobel_abs_y = cv2.convertScaleAbs(sobel_y)
        edged = cv2.addWeighted(sobel_abs_x, 0.5, sobel_abs_y, 0.5, 0)

        lines = cv2.HoughLinesP(edged, 1, np.pi / 180, 10, 5, 5)
        return lines

    @staticmethod
    def _find_best_matched_line(thetaA0, thetaB0, tolerance, vectors, matched=None, start_index=0):
        if matched is not None:
            matched_distance, matched_length, matched_thetaA, matched_thetaB, matched_coord = matched
            matched_angle = abs(np.pi / 2 - matched_thetaB)

        for i in xrange(start_index, len(vectors)):
            distance, length, thetaA, thetaB, coord = vectors[i]

            if (thetaA0 is None or abs(thetaA - thetaA0) <= tolerance) and \
                    (thetaB0 is None or abs(thetaB - thetaB0) <= tolerance):

                if matched is None:
                    matched = vectors[i]
                    matched_distance, matched_length, matched_thetaA, matched_thetaB, matched_coord = matched
                    matched_angle = abs(np.pi / 2 - matched_thetaB)
                    continue

                heading_angle = abs(np.pi / 2 - thetaB)

                if heading_angle > matched_angle:
                    continue
                if heading_angle < matched_angle:
                    matched = vectors[i]
                    matched_distance, matched_length, matched_thetaA, matched_thetaB, matched_coord = matched
                    matched_angle = abs(np.pi / 2 - matched_thetaB)
                    continue
                if distance < matched_distance:
                    continue
                if distance > matched_distance:
                    matched = vectors[i]
                    matched_distance, matched_length, matched_thetaA, matched_thetaB, matched_coord = matched
                    matched_angle = abs(np.pi / 2 - matched_thetaB)
                    continue
                if length < matched_length:
                    continue
                if length > matched_length:
                    matched = vectors[i]
                    matched_distance, matched_length, matched_thetaA, matched_thetaB, matched_coord = matched
                    matched_angle = abs(np.pi / 2 - matched_thetaB)
                    continue

        return matched

    @staticmethod
    def mask_frame_color(frame, min_values, max_values):
        """
        Mask a frame by RGB color
        """

        min_th_ok = np.all(frame > min_values, axis=2)
        max_th_ok = np.all(frame < max_values, axis=2)

        out = np.logical_and(min_th_ok, max_th_ok)

        return out

    @staticmethod
    def binarize(img):
        """
        Convert an input frame to a binary image
        :param img: input color frame
        :param verbose: if True, show intermediate results
        :return: binarized frame
        """
        B_min = np.array([100, 0, 0])
        B_max = np.array([255, 150, 150])

        G_min = np.array([0, 100, 0])
        G_max = np.array([150, 255, 150])

        R_min = np.array([0, 0, 100])
        R_max = np.array([150, 150, 255])

        h, w = img.shape[:2]
        binary = np.zeros(shape=(h, w), dtype=np.uint8)

        B_mask = ImageProcessor.mask_frame_color(img, B_min, B_max)
        binary = np.logical_or(binary, B_mask)

        G_mask = ImageProcessor.mask_frame_color(img, G_min, G_max)
        binary = np.logical_or(binary, G_mask)

        R_mask = ImageProcessor.mask_frame_color(img, R_min, R_max)
        binary = np.logical_or(binary, R_mask)

        # get Sobel binary mask (thresholded gradients)
        # sobel_mask = mask_frame_sobel(img, kernel_size=9)
        # binary = np.logical_or(binary, sobel_mask)

        # apply a light morphology to "fill the gaps" in the binary image
        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

        return closing

    @staticmethod
    def check_recovery_direction(img):
        """
        check if we should move forward or backward. also check if we hit right wall or left wall
        """
        move_forward = False
        px = 0
        # use only a 10-pixel slice to enhance performance. 150~160 is based on experience.
        part = slice(150, 160)
        section = img[part, :]

        # turn RGB images into binary images
        section = ImageProcessor.binarize(section)
        _y, _x = np.where(section == 1)
        len_x = len(_x)
        # if there is no obstacle/wall ahead, move forward, else backward
        if len_x > 3000:
            move_forward = True

        # if px > 160, it should be left wall. otherwise, right wall.
        if len_x > 500:
            px = np.mean(_x)

        return move_forward, px

    @staticmethod
    def find_steering_angle_by_line(img, last_steering_angle, debug=True):
        steering_angle = 0.0
        lines = ImageProcessor.find_lines(img)

        if lines is None:
            return steering_angle

        image_height = img.shape[0]
        image_width = img.shape[1]
        camera_x = image_width / 2
        camera_y = image_height
        vectors = []

        for line in lines:
            for x1, y1, x2, y2 in line:
                thetaA = math.atan2(abs(y2 - y1), (x2 - x1))
                thetaB1 = math.atan2(abs(y1 - camera_y), (x1 - camera_x))
                thetaB2 = math.atan2(abs(y2 - camera_y), (x2 - camera_x))
                thetaB = thetaB1 if abs(np.pi / 2 - thetaB1) < abs(np.pi / 2 - thetaB2) else thetaB2

                length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                distance = min(math.sqrt((x1 - camera_x) ** 2 + (y1 - camera_y) ** 2),
                               math.sqrt((x2 - camera_x) ** 2 + (y2 - camera_y) ** 2))

                vectors.append((distance, length, thetaA, thetaB, (x1, y1, x2, y2)))

                if debug:
                    # draw the edges
                    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # the line of the shortest distance and longer length will be the first choice
        vectors.sort(lambda a, b: cmp(a[0], b[0]) if a[0] != b[0] else -cmp(a[1], b[1]))

        best = vectors[0]
        best_distance, best_length, best_thetaA, best_thetaB, best_coord = best
        tolerance = np.pi / 180.0 * 10.0

        best = ImageProcessor._find_best_matched_line(best_thetaA, None, tolerance, vectors, matched=best,
                                                      start_index=1)
        best_distance, best_length, best_thetaA, best_thetaB, best_coord = best

        if debug:
            # draw the best line
            cv2.line(img, best_coord[:2], best_coord[2:], (0, 255, 255), 2)

        if abs(best_thetaB - np.pi / 2) <= tolerance and abs(best_thetaA - best_thetaB) >= np.pi / 4:
            print("*** sharp turning")
            best_x1, best_y1, best_x2, best_y2 = best_coord
            f = lambda x: int(
                ((float(best_y2) - float(best_y1)) / (float(best_x2) - float(best_x1)) * (x - float(best_x1))) + float(
                    best_y1))
            left_x, left_y = 0, f(0)
            right_x, right_y = image_width - 1, f(image_width - 1)

            if left_y < right_y:
                best_thetaC = math.atan2(abs(left_y - camera_y), (left_x - camera_x))

                if debug:
                    # draw the last possible line
                    cv2.line(img, (left_x, left_y), (camera_x, camera_y), (255, 128, 128), 2)
                    cv2.line(img, (left_x, left_y), (best_x1, best_y1), (255, 128, 128), 2)
            else:
                best_thetaC = math.atan2(abs(right_y - camera_y), (right_x - camera_x))

                if debug:
                    # draw the last possible line
                    cv2.line(img, (right_x, right_y), (camera_x, camera_y), (255, 128, 128), 2)
                    cv2.line(img, (right_x, right_y), (best_x1, best_y1), (255, 128, 128), 2)

            steering_angle = best_thetaC
        else:
            steering_angle = best_thetaB

        if (steering_angle - np.pi / 2) * (last_steering_angle - np.pi / 2) < 0:
            last = ImageProcessor._find_best_matched_line(None, last_steering_angle, tolerance, vectors)

            if last:
                last_distance, last_length, last_thetaA, last_thetaB, last_coord = last
                steering_angle = last_thetaB

                if debug:
                    # draw the last possible line
                    cv2.line(img, last_coord[:2], last_coord[2:], (255, 128, 128), 2)

        if debug:
            # draw the steering direction
            r = 60
            x = image_width / 2 + int(r * math.cos(steering_angle))
            y = image_height - int(r * math.sin(steering_angle))
            cv2.line(img, (image_width / 2, image_height), (x, y), (255, 0, 255), 2)
            logit("line angle: %0.2f, steering angle: %0.2f, last steering angle: %0.2f" % (
            ImageProcessor.rad2deg(best_thetaA), ImageProcessor.rad2deg(np.pi / 2 - steering_angle),
            ImageProcessor.rad2deg(np.pi / 2 - last_steering_angle)))

        return (np.pi / 2 - steering_angle)

    @staticmethod
    def find_steering_angle_by_color(img, last_steering_angle, debug=True):
        r, g, b = cv2.split(img)
        image_height = img.shape[0]
        image_width = img.shape[1]
        camera_x = image_width / 2
        image_sample = slice(0, int(image_height * 0.2))
        sr, sg, sb = r[image_sample, :], g[image_sample, :], b[image_sample, :]
        track_list = [sr, sg, sb]
        tracks = map(lambda x: len(x[x > 20]), [sr, sg, sb])
        tracks_seen = filter(lambda y: y > 50, tracks)

        if len(tracks_seen) == 0:
            return 0.0

        maximum_color_idx = np.argmax(tracks, axis=None)
        _target = track_list[maximum_color_idx]
        _y, _x = np.where(_target == 255)
        px = np.mean(_x)
        steering_angle = math.atan2(image_height, (px - camera_x))

        if debug:
            # draw the steering direction
            r = 60
            x = image_width / 2 + int(r * math.cos(steering_angle))
            y = image_height - int(r * math.sin(steering_angle))
            cv2.line(img, (image_width / 2, image_height), (x, y), (255, 0, 255), 2)
            logit("steering angle: %0.2f, last steering angle: %0.2f" % (
            ImageProcessor.rad2deg(steering_angle), ImageProcessor.rad2deg(np.pi / 2 - last_steering_angle)))

        return (np.pi / 2 - steering_angle) * 2.0

    @staticmethod
    def find_min_horizon_line_y(img, slice_shift_unit, percentage_image_pixels, debug=True):
        r, g, b = cv2.split(img)
        image_height = img.shape[0]
        image_width = img.shape[1]
        camera_x = image_width / 2
        horizon_line_y = 0
        while horizon_line_y < image_height:
            sr, sg, sb = (r[horizon_line_y:horizon_line_y + slice_shift_unit, :],
                          g[horizon_line_y:horizon_line_y + slice_shift_unit, :],
                          b[horizon_line_y:horizon_line_y + slice_shift_unit, :])
            slice_images_pixels = sg.shape[0] * sg.shape[1]
            tracks = map(lambda x: len(x[x == 0]), [sr, sg, sb])  # r,g,b = (0,0,0)
            tracks_seen = filter(lambda y: float(y) / slice_images_pixels < percentage_image_pixels, tracks)
            if debug and len(tracks_seen) != 0:
                print(horizon_line_y)
                print(tracks)
            if len(tracks_seen) != 0:
                break
            horizon_line_y += slice_shift_unit
        return horizon_line_y

    @staticmethod
    def find_trajectory_line(img, trajectory, lines, base_x, base_y, max_trajectory,
                             min_horizon_line_y, debug=True):
        if max_trajectory == 0 or len(lines) <= 5: return trajectory
        vectors = []
        del_idx_list = []
        for idx, line in enumerate(lines):
            for x1, y1, x2, y2 in line:
                thetaA = math.atan2(abs(y2 - y1), (x2 - x1))
                thetaB1 = math.atan2(abs(y1 - base_y), (x1 - base_x))
                thetaB2 = math.atan2(abs(y2 - base_y), (x2 - base_x))
                thetaB = thetaB1 if abs(np.pi / 2 - thetaB1) < abs(np.pi / 2 - thetaB2) else thetaB2
                if (y1 > base_y
                    or y2 > base_y
                    or (y1 < min_horizon_line_y and y2 < min_horizon_line_y)
                    or float(thetaA) < 0.017453292519943295
                    ):
                    del_idx_list.append(idx)
                    continue
                length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                distance = min(math.sqrt((x1 - base_x) ** 2 + (y1 - base_y) ** 2),
                               math.sqrt((x2 - base_x) ** 2 + (y2 - base_y) ** 2))
                vectors.append((distance, length, thetaA, thetaB, (x1, y1, x2, y2), idx))
        if len(vectors) == 0: return trajectory

        # the line of the shortest distance (and longer length will be the first choice)
        vectors.sort(lambda a, b: cmp(a[0], b[0]))  # if a[0] != b[0] else -cmp(a[1], b[1]))
        best = vectors[0]
        best_distance, best_length, best_thetaA, best_thetaB, best_coord, idx = best
        if (len(trajectory) < 1
            or (len(trajectory) >= 1
                and best_distance < 30.
                # and abs(ImageProcessor.rad2deg(thetaA) - ImageProcessor.rad2deg(trajectory[-1][2])) < 5.0
                )
            ):
            trajectory.append(best)
            cv2.line(img, best_coord[:2], best_coord[2:], (255, 255, 0), 5)  # (0, 255, 255)
            if debug:
                logit("*** trajectory line --- distance: %f, length: %f, line: %s, thetaA: %d, thetaB: %d" % (
                best_distance, best_length, best_coord, ImageProcessor.rad2deg(best_thetaA), ImageProcessor.rad2deg(best_thetaB)))
            x1, y1, x2, y2 = best_coord
            base_x, base_y = (x1, y1) if y1 < y2 else (x2, y2)
            max_trajectory -= 1
            del_idx_list.append(idx)
            new_lines = np.delete(lines, del_idx_list, 0)
            return ImageProcessor.find_trajectory_line(img, trajectory, new_lines, base_x, base_y, max_trajectory, min_horizon_line_y,
                                        debug)
        else:
            return trajectory

    @staticmethod
    def find_trajectory_points(img, trajectory, image_height, min_horizon_line_y, n_points, debug=True):
        points = []
        unit_shift_y = (image_height - min_horizon_line_y) / n_points
        shift_horizon_line_y = image_height
        while n_points >= -1:
            shift_horizon_line_y = shift_horizon_line_y - unit_shift_y
            for line in trajectory:
                thetaA = line[2]
                x1, y1, x2, y2 = line[4]
                near_camera_x, near_camera_y, far_camera_x, far_camera_y = (x2, y2, x1, y1) if min(y1,y2) == y1 else (
                x1, y1, x2, y2)
                if (
                            ((near_camera_y > shift_horizon_line_y and far_camera_y < shift_horizon_line_y)
                             or (far_camera_y - shift_horizon_line_y) < 1)
                        and float(thetaA) > 0.017453292519943295
                        and max(y1, y2) >= shift_horizon_line_y
                ):
                    ajancent = b = near_camera_y - shift_horizon_line_y
                    theta = x = (math.pi / 2) - thetaA if far_camera_x >= near_camera_x else thetaA - (math.pi / 2)
                    opposite = a = math.tan(x) * b
                    p_x = int(near_camera_x + a)
                    p_y = int(near_camera_y - b)
                    points.append((p_x, p_y))
                    cv2.circle(img, (p_x, p_y), 10, (0, 255, 255), -1)
                    if debug:
                        print("shift_horizon_line_y: %d, thetaA: %f" % (shift_horizon_line_y, rad2deg(thetaA)))
                        print("               _x,_y: %d, %d" % (p_x, p_y))
                        print("         nx,ny,fx,fy: %d, %d %d %d" % (
                        near_camera_x, near_camera_y, far_camera_x, far_camera_y))
                        print("theta,ajancent,opposite: %f, %f, %f" % (rad2deg(theta), ajancent, opposite))
                    break
                else:
                    continue
            n_points -= 1
        return points

    @staticmethod
    def find_center_points(img):
        (_, cnts, _) = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        center_poins = []
        if not cnts:
            return center_poins

        for c in cnts:
            moments = cv2.moments(c)
            m00 = moments['m00']
            if m00 != 0:
                centroid_x = int(moments['m10'] / m00)  # Take X coordinate
                centroid_y = int(moments['m01'] / m00)  # Take Y coordinate
                center_poins.append((centroid_x, centroid_y))
        return center_poins

    @staticmethod
    def get_distance(point_1, point_2):
        return math.sqrt((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2)

    @staticmethod
    def get_slope(pre_point, current_point):
        if current_point[1] == pre_point[1]:
            return
        return float(current_point[0] - pre_point[0]) / float(current_point[1] - pre_point[1])

    @staticmethod
    def filter_points(candidate_points, pre_point):
        if not candidate_points:
            return
        result_point = None
        min_distance_diff = -1
        for point in candidate_points:
            current_distance = ImageProcessor.get_distance(point, pre_point)
            if current_distance < min_distance_diff or min_distance_diff == -1:
                min_distance_diff = current_distance
                result_point = point
        return result_point

    @staticmethod
    def find_mpc_ref_points(img, init_point, lower_hsv, upper_hsv, debug=False):
        image_height = img.shape[0]
        shift = 10
        step_scale = 5

        ref_points = []
        pre_point = init_point
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
        if debug:
            ImageProcessor.show_image(mask, 'mask')
        while image_height > int(image_height * 0.2):
            start_y = (image_height - shift)
            src = mask[start_y: start_y + 30]
            image_height -= shift
            center_points = ImageProcessor.find_center_points(src)
            point = ImageProcessor.filter_points(candidate_points=center_points,
                                                 pre_point=pre_point)
            if point:
                point = (point[0], point[1] + start_y)

                # If there's a point with the same y-value
                if ref_points and point[1] == ref_points[-1][1]:
                    point = (point[0], point[1] - 1)

                ref_points.append(point)
                pre_point = point

            image_height -= shift
            if step_scale > step_scale:
                step_scale -= step_scale
        return ref_points

    @staticmethod
    def mpc_control_by_color(img, lower_hsv, upper_hsv, debug=False):
        ref_points = ImageProcessor.find_mpc_ref_points(img=img,
                                                        init_point=(img.shape[1] / 2, 0),
                                                        lower_hsv=lower_hsv,
                                                        upper_hsv=upper_hsv,
                                                        debug=debug)

        return ref_points

    cache_points = []
    @staticmethod
    def mpc_control_by_line(img):
        # img is track_img
        image_height = img.shape[0]
        image_width = img.shape[1]
        camera_x = image_width / 2
        camera_y = image_height
        lines = ImageProcessor.find_lines(img)
        NUM_POINTS = 6
        min_horizon_line_y = ImageProcessor.find_min_horizon_line_y(img, 5, .8, False)
        trajectory = []
        trajectory = ImageProcessor.find_trajectory_line(img, trajectory, lines, camera_x, camera_y, 30,
                                          min_horizon_line_y, debug=True)
        points = ImageProcessor.find_trajectory_points(img, trajectory, image_height, min_horizon_line_y, NUM_POINTS, debug=False)
        return points
        # if len(points) <= 3:
        #     return ImageProcessor.cache_points
        # else:
        #     ImageProcessor.cache_points = points
        #     return points


class AutoDrive(object):
    STEERING_PID_Kp = 0.3
    STEERING_PID_Ki = 0.01
    STEERING_PID_Kd = 0.1
    STEERING_PID_max_integral = 10
    THROTTLE_PID_Kp = 0.02
    THROTTLE_PID_Ki = 0.005
    THROTTLE_PID_Kd = 0.02
    THROTTLE_PID_max_integral = 0.5
    MAX_STEERING_HISTORY = 3
    MAX_THROTTLE_HISTORY = 3
    DEFAULT_SPEED = 0.5

    debug = False

    def __init__(self, mpc_library_path, mpc_settings_path, record_folder=None, do_sign_detection=True):
        self._record_folder = record_folder
        self._steering_pid = PID(Kp=self.STEERING_PID_Kp, Ki=self.STEERING_PID_Ki, Kd=self.STEERING_PID_Kd,
                                 max_integral=self.STEERING_PID_max_integral)
        self._throttle_pid = PID(Kp=self.THROTTLE_PID_Kp, Ki=self.THROTTLE_PID_Ki, Kd=self.THROTTLE_PID_Kd,
                                 max_integral=self.THROTTLE_PID_max_integral)
        self._throttle_pid.assign_set_point(self.DEFAULT_SPEED)
        self._steering_history = []
        self._throttle_history = []
        self._last_steering_history = 0
        self._last_throttle_history = 0

        self._mpc_model = MPC(mpc_library_path, mpc_settings_path, debug=self.debug)

        self.angle_before_hit_wall = 0
        self.hit_wall = False

        self._crash = 0
        self._record_images = []
        self._recover_mode = 0
        self._recover_steering = 0.0

    def on_dashboard(self, src_img, last_steering_angle, speed, throttle, info):
        if USE_MPC:
            return self.on_dashboard_mpc(src_img, last_steering_angle, speed, throttle, info)
        else:
            return self.on_dashboard_pid(src_img, last_steering_angle, speed, throttle, info)

    def on_dashboard_mpc(self, src_img, last_steering_angle, speed, throttle, info):

        if REF_LINE == MPC_MODE.mpc_control_by_color:
            r_upper_bound = (10, 255, 255)
            r_lower_bound = (0, 43, 46)
            b_upper_bound = (150, 255, 255)
            b_lower_bound = (90, 50, 50)
            ref_points = ImageProcessor.mpc_control_by_color(cv2.blur(src_img, (10, 10)), b_lower_bound, b_upper_bound,
                                                             self.debug)
        else:
            # img = cv2.blur(src_img, (10, 10))
            # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # kernel_size = 5
            # blurred = cv2.GaussianBlur(hsv, (kernel_size, kernel_size), 0)
            # lower_hsv = b_lower_bound = (90, 50, 50)
            # upper_hsv = b_upper_bound = (150, 255, 255)
            # mask = cv2.inRange(blurred, np.array(lower_hsv), np.array(upper_hsv))
            # track_img = cv2.bitwise_and(src_img, src_img, mask=mask)
            track_img = ImageProcessor.preprocess(src_img)
            ref_points = ImageProcessor.mpc_control_by_line(track_img)
            src_img = copy.copy(track_img)

        if self.debug:
            print("ref_points: %s " % ref_points)

        image_height = src_img.shape[0]
        image_width = src_img.shape[1]
        camera_point = (image_width / 2, 0)
        mpc_ref_points = []
        for point in ref_points:
            mpc_ref_point = (image_height - point[1], camera_point[0] - point[0])
            mpc_ref_points.append(mpc_ref_point)
        point_scale = 5
        predict_result = self._mpc_model.run([point[0] * 1.0 / point_scale for point in mpc_ref_points],
                                             [point[1] * 1.0 / point_scale for point in mpc_ref_points],
                                             speed)
        result_dict = json.loads(predict_result)

        if self.debug:
            for i in range(len(ref_points)):
                cv2.circle(src_img, ref_points[i], 5, (0, 0, 255))
                cv2.putText(src_img, str(mpc_ref_points[i]).decode(), ref_points[i], cv2.FONT_HERSHEY_COMPLEX, 0.3, 255)
            mpc_x_list = result_dict['mpc_x']
            mpc_y_list = result_dict['mpc_y']
            for i in range(len(mpc_x_list)):
                x = camera_point[0] - mpc_y_list[i] * point_scale
                y = image_height - mpc_x_list[i] * point_scale
                cv2.circle(src_img, (int(x), int(y)), 5, (0, 255, 0))
            ImageProcessor.show_image(src_img, "source")

        # Handle crash accident
        crashed = self.crash_detector(speed)
        if crashed:
            new_steering_angle, new_throttle = self.crash_recover(src_img, speed)
        else:
            if predict_result:
                self._last_steering_history = result_dict['steering_angle']
                self._last_throttle_history = result_dict['throttle']
            new_steering_angle, new_throttle = self._last_steering_history, self._last_throttle_history

        self.record_image(src_img)

        return new_steering_angle, new_throttle

    def on_dashboard_pid(self, src_img, last_steering_angle, speed, throttle, info):
        track_img = current_angle = None
        if REF_LINE == MPC_MODE.mpc_control_by_color:
            track_img = ImageProcessor.preprocess(src_img)
            current_angle = ImageProcessor.find_steering_angle_by_color(track_img, last_steering_angle,
                                                                        debug=self.debug)
            blur = cv2.blur(src_img, (10, 10))
            r_upper_bound = (10, 255, 255)
            r_lower_bound = (0, 43, 46)
            b_upper_bound = (150, 255, 255)
            b_lower_bound = (90, 50, 50)
            ImageProcessor.mpc_control_by_color(blur, r_lower_bound, r_upper_bound, self.debug)
        else:
            track_img = ImageProcessor.preprocess(src_img, .25)
            current_angle = ImageProcessor.find_steering_angle_by_line(track_img, last_steering_angle, debug=self.debug)
            track_img = ImageProcessor.preprocess(src_img, .25)
            ImageProcessor.mpc_control_by_line(track_img)

        steering_angle = self._steering_pid.update(-current_angle)
        throttle = self._throttle_pid.update(speed)

        if self.debug:
            ImageProcessor.show_image(src_img, "source")
            ImageProcessor.show_image(track_img, "track")
            # logit("steering PID: %0.2f (%0.2f) => %0.2f (%0.2f)" % (current_angle, ImageProcessor.rad2deg(current_angle), steering_angle, ImageProcessor.rad2deg(steering_angle)))
            # logit("throttle PID: %0.4f => %0.4f" % (speed, throttle))
            # logit("info: %s" % repr(info))

        if self._record_folder:
            suffix = "-deg%0.3f" % (ImageProcessor.rad2deg(steering_angle))
            ImageProcessor.save_image(self._record_folder, src_img, prefix="cam", suffix=suffix)
            ImageProcessor.save_image(self._record_folder, track_img, prefix="trk", suffix=suffix)

        # smooth the control signals
        self._steering_history.append(steering_angle)
        self._steering_history = self._steering_history[-self.MAX_STEERING_HISTORY:]
        self._throttle_history.append(throttle)
        self._throttle_history = self._throttle_history[-self.MAX_THROTTLE_HISTORY:]

        # Handle crash accident
        crashed = self.crash_detector(speed)
        if crashed:
            new_steering_angle, new_throttle = self.crash_recover(src_img, speed)
        else:
            new_steering_angle = sum(self._steering_history) / self.MAX_STEERING_HISTORY
            new_throttle = sum(self._throttle_history) / self.MAX_THROTTLE_HISTORY

        self.record_image(src_img)

        return new_steering_angle, new_throttle

    def crash_detector(self, speed):
        """
        detect if a crash occurred
        """
        crash = False

        # consider speed < 0.1 sort of crash
        if speed < 0.1:
            self._crash = self._crash + 1
        elif self._recover_mode == 0 and speed >= 0.3:
            self._crash = 0

        # it can be 5 or even shorter, depending on your bot behavior
        if self._crash >= 8:
            crash = True

        return crash

    def record_image(self, src_img):
        """
        record last 100 images for replay. used to detect when a crash hit right wall or left wall
        """

        self._record_images.append(src_img)
        self._record_images = self._record_images[-100:]

    def crash_recover(self, src_img, speed):
        """
        decide recover steering angle and throttle based on the image replay
        """
        self._recover_mode = 1

        # skip calculating recover steering angle if we already have it
        if self._recover_steering == 0.0:
            for i in range(1, len(self._record_images)):
                # replay the recorded images backward. check if the car hit right wall or left wall
                _, px = ImageProcessor.check_recovery_direction(self._record_images[-i])
                # px > 165 means left wall. px < 155 means right wall
                if px > 165:
                    self._recover_steering = -40
                    break
                elif 0 < px < 155:
                    self._recover_steering = 40
                    break
        # check if we should move forward or backward
        move_forward, px = ImageProcessor.check_recovery_direction(src_img)
        if move_forward:
            # print("recovery end")
            recover_steering = -self._recover_steering
            recover_throttle = 0.2
            self._recover_mode = 0
            self._crash = 0
            self._recover_steering = 0.0
        else:
            # print("recovering")
            recover_steering = self._recover_steering
            recover_throttle = -1

        return recover_steering, recover_throttle


class MpcCar(Car):
    MAX_STEERING_ANGLE = 40.0

    def __init__(self, driver):
        self._driver = driver

    def on_dashboard(self, dashboard):
        # normalize the units of all parameters
        last_steering_angle = np.pi / 2 - float(dashboard["steering_angle"]) / 180.0 * np.pi
        throttle = float(dashboard["throttle"])
        brake = float(dashboard["brakes"])
        speed = float(dashboard["speed"])
        img = ImageProcessor.bgr2rgb(np.asarray(Image.open(BytesIO(base64.b64decode(dashboard["image"])))))
        del dashboard["image"]
        # print(datetime.now(), dashboard)
        total_time = float(dashboard["time"])
        elapsed = total_time

        info = {
            "lap": int(dashboard["lap"]) if "lap" in dashboard else 0,
            "elapsed": elapsed,
            "status": int(dashboard["status"]) if "status" in dashboard else 0,
        }
        new_steering_angle, new_throttle = self._driver.on_dashboard(img, last_steering_angle, speed, throttle, info)

        new_steering_angle = min(max(ImageProcessor.rad2deg(new_steering_angle), -MpcCar.MAX_STEERING_ANGLE),
                                 MpcCar.MAX_STEERING_ANGLE)

        return new_steering_angle, new_throttle


def create_mpc_driver(lib_dir, ref_line='color', record_folder=None, do_sign_detection=True):
    global REF_LINE

    print("MPC mode is %s" % ref_line)
    REF_LINE = MPC_MODE.mpc_control_by_color if (ref_line == 'color') else MPC_MODE.mpc_control_by_line

    mpc_library_path, mpc_settings_path = get_model_path(lib_dir)
    return AutoDrive(mpc_library_path, mpc_settings_path, record_folder, do_sign_detection)


def get_model_path(root_dir):
    if platform == "linux" or platform == "linux2":
        mpc_library_path = os.path.join(root_dir, "./libmpc_linux.so")
    elif platform == "darwin":
        mpc_library_path = os.path.join(root_dir, "./libmpc_mac.so")
    else:
        mpc_library_path = None
    mpc_settings_path = os.path.join(root_dir, "./mpc_config.json")

    return mpc_library_path, mpc_settings_path


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='AutoDriveBot')
    parser.add_argument(
        'record',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder to record the images.'
    )
    parser.add_argument(
        'mpc_mode',
        type=str,
        nargs='?',
        default='color',
        help='Specify the mpc mode by color or line.'
    )

    args = parser.parse_args()

    # Input arguments
    if args.record:
        if not os.path.exists(args.record):
            os.makedirs(args.record)
        logit("Start recording images to %s..." % args.record)

    # Create car with MPC driver
    driver = create_mpc_driver(lib_dir=os.getcwd(), ref_line=args.mpc_mode, record_folder=args.record)
    car = MpcCar(driver=driver)

    sio = socketio.Server()

    def send_control(steering_angle, throttle):
        sio.emit(
            "steer",
            data={
                'steering_angle': str(steering_angle),
                'throttle': str(throttle)
            },
            skip_sid=True)

    def send_restart():
        sio.emit(
            "restart",
            data={},
            skip_sid=True)

    @sio.on('telemetry')
    def telemetry(sid, dashboard):
        if dashboard:
            steering_angle, throttle = car.on_dashboard(dashboard)
            send_control(steering_angle, throttle)
        else:
            sio.emit('manual', data={}, skip_sid=True)

    @sio.on('connect')
    def connect(sid, environ):
        send_control(0, 0)

    app = socketio.Middleware(sio, Flask(__name__))
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

# vim: set sw=4 ts=4 et :

