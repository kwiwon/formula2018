#!/usr/bin/python
# -*- coding: utf-8 -*-

#!env python
#
# Auto-driving Bot
#
# Revision:      v1.2
# Released Date: Aug 20, 2018
#

from time import time
from PIL  import Image
from io   import BytesIO

#import datetime
import os
import cv2
import math
import numpy as np
import base64
import logging
import collections
import sys

import utils_json as utils
#load our saved model
from keras.models import load_model as ks_load_model

LOG_LEVEL = logging.DEBUG

logger = logging.getLogger('bot')
hdlr = logging.FileHandler('../log/bot.log', mode='w')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(LOG_LEVEL)

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
ch.setLevel(LOG_LEVEL)

class ConstValue(object):
    DEBUG               = True
    # image size
    IMAGE_WIDTH         = 320
    IMAGE_HEIGHT        = 240
    # image crop range
    IMAGE_SIGN_HEIGHT   = 0.35  #0~0.35
    IMAGE_TRACK_HEIGHT  = 0.50   #0.5~1
    # track color code in COLOR_BGR2GRAY
    TRACK_GRAY_BLUE     = 29
    TRACK_GRAY_RED      = 76
    TRACK_GRAY_GREEN    = 150
    TRACK_GRAY_WALL     = 0 #black, yellow

    IMAGE_WALL_START    = 0.6 #0.6~1

    MIN_SPEED           = 0.1
    STUCK_COUNTER       = 10
    

class SignItem(object):

    def __init__(self, img_path, direction):
        self._sign_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        self._direction = direction

    @property
    def image(self):
        return self._sign_img
    @property
    def direction(self):
        return self._direction
    
    
class Sign(object):

    DIRECTION_LEFT  = 'left'
    DIRECTION_RIGHT = 'right'

    def __init__(self):
        self.sign_list = []
        self.sign_list.append(SignItem('sign-02/sign-02-med.jpg', Sign.DIRECTION_RIGHT))
        self.sign_list.append(SignItem('sign-02/sign-02-med2.jpg', Sign.DIRECTION_RIGHT))
        self.sign_list.append(SignItem('sign-02/sign-02-med3.jpg', Sign.DIRECTION_RIGHT))
        self.sign_list.append(SignItem('sign-02/sign-02-small.jpg', Sign.DIRECTION_RIGHT))
        self.sign_list.append(SignItem('sign-04/sign-04-med.jpg', Sign.DIRECTION_LEFT))
        self.sign_list.append(SignItem('sign-04/sign-04-small.jpg', Sign.DIRECTION_LEFT))
        self.sign_list.append(SignItem('sign-04/sign-04-med2.jpg', Sign.DIRECTION_LEFT))
        self.sign_list.append(SignItem('sign-04/sign-04-small2.jpg', Sign.DIRECTION_LEFT))
        self.sign_list.append(SignItem('sign-04/sign-04-med3.jpg', Sign.DIRECTION_LEFT))
        self.sign_list.append(SignItem('sign-04/sign-04-small3.jpg', Sign.DIRECTION_LEFT))
        self.sign_list.append(SignItem('sign-04/sign-04-med4.jpg', Sign.DIRECTION_LEFT))
        self.sign_list.append(SignItem('sign-07/sign-07-med.jpg', Sign.DIRECTION_RIGHT))
        self.sign_list.append(SignItem('sign-07/sign-07-small.jpg', Sign.DIRECTION_RIGHT))
        self.sign_list.append(SignItem('sign-08/sign-08-med.jpg', Sign.DIRECTION_LEFT))
        self.sign_list.append(SignItem('sign-08/sign-08-small.jpg', Sign.DIRECTION_LEFT))
        self.sign_list.append(SignItem('sign-08/sign-08-med2.jpg', Sign.DIRECTION_LEFT))
        self.sign_list.append(SignItem('sign-08/sign-08-small2.jpg', Sign.DIRECTION_LEFT))

    # return find, direction, 
    def find_sign(self, img_src, img_draw):
        for signItem in self.sign_list:
            w = signItem.image.shape[1]
            h = signItem.image.shape[0]
            #same direction: 'cv2.TM_CCOEFF', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF_NORMED'
            #different direction: 'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_SQDIFF'
            # methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            #             'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
            methods = ['cv2.TM_CCOEFF_NORMED']
            for meth in methods:
                img = img_src.copy()
                template = signItem.image.copy()
                method = eval(meth)
             
                # Apply template Matching
                res = cv2.matchTemplate(img,template,method)
                threshold = 0.7
                if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                    loc = np.where(res <= threshold)
                else:
                    loc = np.where(res >= threshold)

                pt_list = []
                for pt in zip(*loc[::-1]): 
                    pt_list.append(pt)
                top_left_np = np.array(pt_list)

                if top_left_np.size>0:
                    top_left = (top_left_np.min(0)[0], top_left_np.min(0)[1])
                    bottom_right = (top_left[0]+w, top_left[1]+h)
                    cv2.rectangle(img_draw, top_left, bottom_right, (0,255,0), 3)
                    cv2.putText(img_draw, signItem.direction, top_left, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,255,0), 1, cv2.LINE_AA)
                    return True, signItem.direction
        # can't find sign
        return False, None

def logit(msg):
    if ConstValue.DEBUG:
        logger.debug(msg)


class PID:
    def __init__(self, Kp, Ki, Kd, max_integral, min_interval = 0.001, set_point = 0.0, last_time = None):
        self._Kp           = Kp
        self._Ki           = Ki
        self._Kd           = Kd
        self._min_interval = min_interval
        self._max_integral = max_integral

        self._set_point    = set_point
        self._last_time    = last_time if last_time is not None else time()
        self._p_value      = 0.0
        self._i_value      = 0.0
        self._d_value      = 0.0
        self._d_time       = 0.0
        self._d_error      = 0.0
        self._last_error   = 0.0
        self._output       = 0.0


    def update(self, cur_value, cur_time = None):
        if cur_time is None:
            cur_time = time()

        error   = self._set_point - cur_value
        d_time  = cur_time - self._last_time
        d_error = error - self._last_error

        if d_time >= self._min_interval:
            self._p_value   = error
            self._i_value   = min(max(error * d_time, -self._max_integral), self._max_integral)
            self._d_value   = d_error / d_time if d_time > 0 else 0.0
            self._output    = self._p_value * self._Kp + self._i_value * self._Ki + self._d_value * self._Kd

            self._d_time     = d_time
            self._d_error    = d_error
            self._last_time  = cur_time
            self._last_error = error

        return self._output

    def reset(self, last_time = None, set_point = 0.0):
        self._set_point    = set_point
        self._last_time    = last_time if last_time is not None else time()
        self._p_value      = 0.0
        self._i_value      = 0.0
        self._d_value      = 0.0
        self._d_time       = 0.0
        self._d_error      = 0.0
        self._last_error   = 0.0
        self._output       = 0.0

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
    def show_image(img, name = "image", scale = 1.0):
        if scale and scale != 1.0:
            img = cv2.resize(img, newsize, interpolation=cv2.INTER_CUBIC) 

        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(name, img)
        cv2.waitKey(1)


    @staticmethod
    def save_image(folder, img, prefix = "img", suffix = ""):
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
        adjustment = min(255.0/img.max(), 3.0)
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
    def _crop_image(img):
        bottom_half_ratios = (ConstValue.IMAGE_TRACK_HEIGHT, 1.0)
        bottom_half_slice  = slice(*(int(x * img.shape[0]) for x in bottom_half_ratios))
        bottom_half        = img[bottom_half_slice, :, :]
        return bottom_half


    @staticmethod
    def preprocess(img):
        img = ImageProcessor._crop_image(img)
        #img = ImageProcessor._normalize_brightness(img)
        img = ImageProcessor._flatten_rgb(img)
        return img

    @staticmethod
    def preprocessTop(img):
        # crop top 35%, too small is hard to idenity
        top_half_ratios = (0, ConstValue.IMAGE_SIGN_HEIGHT)
        top_slice  = slice(*(int(x * img.shape[0]) for x in top_half_ratios))
        top_half   = img[top_slice, :, :]
        img_src    = cv2.cvtColor(top_half, cv2.COLOR_BGR2HSV)


        struct_element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3));
        img_src = cv2.erode(img_src,struct_element, iterations=1)

        LowerRed1 = np.array([0, 42, 46])
        UpperRed1 = np.array([10, 255, 255])
        img_sign1 = cv2.inRange(img_src, LowerRed1, UpperRed1)
        LowerRed2 = np.array([156, 42, 46])
        UpperRed2 = np.array([180, 255, 255])
        img_sign2 = cv2.inRange(img_src, LowerRed2, UpperRed2)
        img_sign = img_sign1 | img_sign2

        final = cv2.erode(img_sign,struct_element, iterations=1)

        final = cv2.bitwise_not(final)

        return final

    @staticmethod
    def find_obs(img_src, img_draw, debug = True):
        img = img_src.copy()
        top_half_ratios = (0.3, 0.7)
        top_slice  = slice(*(int(x * img.shape[0]) for x in top_half_ratios))
        top_half   = img[top_slice, :, :]
        height = int(img_src.shape[0]*0.3)

        img_proc = cv2.cvtColor(top_half, cv2.COLOR_BGR2HSV)
        struct_element = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3));

        #找灰色物體
        LowerGray = np.array([0, 0, 46])
        UpperGray = np.array([180, 43, 210])
        img_gr = cv2.inRange(img_proc, LowerGray, UpperGray)
        
        img_gr = cv2.dilate(img_gr,struct_element, iterations=1)
        img_gr = cv2.erode(img_gr,struct_element, iterations=1)

        im2, contours, hierarchy = cv2.findContours(img_gr,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        _y, _x = np.where(img_gr == 255)
        for c in contours:
            M = cv2.moments(c)

    #because there is no traffic line, so find line is a bad idea
    @staticmethod
    def find_lines(img):
        grayed      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred     = cv2.GaussianBlur(grayed, (3, 3), 0)
        #edged      = cv2.Canny(blurred, 0, 150)

        sobel_x     = cv2.Sobel(blurred, cv2.CV_16S, 1, 0)
        sobel_y     = cv2.Sobel(blurred, cv2.CV_16S, 0, 1)
        sobel_abs_x = cv2.convertScaleAbs(sobel_x)
        sobel_abs_y = cv2.convertScaleAbs(sobel_y)
        edged       = cv2.addWeighted(sobel_abs_x, 0.5, sobel_abs_y, 0.5, 0)

        lines       = cv2.HoughLinesP(edged, 1, np.pi / 180, 10, 5, 5)
        return lines


    @staticmethod
    def _find_best_matched_line(thetaA0, thetaB0, tolerance, vectors, matched = None, start_index = 0):
        if matched is not None:
            matched_distance, matched_length, matched_thetaA, matched_thetaB, matched_coord = matched
            matched_angle = abs(np.pi/2 - matched_thetaB)

        for i in xrange(start_index, len(vectors)):
            distance, length, thetaA, thetaB, coord = vectors[i]

            if (thetaA0 is None or abs(thetaA - thetaA0) <= tolerance) and \
               (thetaB0 is None or abs(thetaB - thetaB0) <= tolerance):
                
                if matched is None:
                    matched = vectors[i]
                    matched_distance, matched_length, matched_thetaA, matched_thetaB, matched_coord = matched
                    matched_angle = abs(np.pi/2 - matched_thetaB)
                    continue

                heading_angle = abs(np.pi/2 - thetaB)

                if heading_angle > matched_angle:
                    continue
                if heading_angle < matched_angle:
                    matched = vectors[i]
                    matched_distance, matched_length, matched_thetaA, matched_thetaB, matched_coord = matched
                    matched_angle = abs(np.pi/2 - matched_thetaB)
                    continue
                if distance < matched_distance:
                    continue
                if distance > matched_distance:
                    matched = vectors[i]
                    matched_distance, matched_length, matched_thetaA, matched_thetaB, matched_coord = matched
                    matched_angle = abs(np.pi/2 - matched_thetaB)
                    continue
                if length < matched_length:
                    continue
                if length > matched_length:
                    matched = vectors[i]
                    matched_distance, matched_length, matched_thetaA, matched_thetaB, matched_coord = matched
                    matched_angle = abs(np.pi/2 - matched_thetaB)
                    continue

        return matched


    @staticmethod
    def find_steering_angle_by_line(img, last_steering_angle, debug = True):
        steering_angle = 0.0
        lines          = ImageProcessor.find_lines(img)

        if lines is None:
            return steering_angle

        image_height = img.shape[0]
        image_width  = img.shape[1]
        camera_x     = image_width / 2
        camera_y     = image_height
        vectors      = []

        for line in lines:
            for x1, y1, x2, y2 in line:
                thetaA   = math.atan2(abs(y2 - y1), (x2 - x1))
                thetaB1  = math.atan2(abs(y1 - camera_y), (x1 - camera_x))
                thetaB2  = math.atan2(abs(y2 - camera_y), (x2 - camera_x))
                thetaB   = thetaB1 if abs(np.pi/2 - thetaB1) < abs(np.pi/2 - thetaB2) else thetaB2

                length   = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                distance = min(math.sqrt((x1 - camera_x) ** 2 + (y1 - camera_y) ** 2),
                               math.sqrt((x2 - camera_x) ** 2 + (y2 - camera_y) ** 2))

                vectors.append((distance, length, thetaA, thetaB, (x1, y1, x2, y2)))

                if debug:
                    # draw the edges
                    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)

        #the line of the shortest distance and longer length will be the first choice
        vectors.sort(lambda a, b: cmp(a[0], b[0]) if a[0] != b[0] else -cmp(a[1], b[1]))

        best = vectors[0]
        best_distance, best_length, best_thetaA, best_thetaB, best_coord = best
        tolerance = np.pi / 180.0 * 10.0

        best = ImageProcessor._find_best_matched_line(best_thetaA, None, tolerance, vectors, matched = best, start_index = 1)
        best_distance, best_length, best_thetaA, best_thetaB, best_coord = best

        if debug:
            #draw the best line
            cv2.line(img, best_coord[:2], best_coord[2:], (0, 255, 255), 2)

        if abs(best_thetaB - np.pi/2) <= tolerance and abs(best_thetaA - best_thetaB) >= np.pi/4:
            print('*** sharp turning')
            best_x1, best_y1, best_x2, best_y2 = best_coord
            f = lambda x: int(((float(best_y2) - float(best_y1)) / (float(best_x2) - float(best_x1)) * (x - float(best_x1))) + float(best_y1))
            left_x , left_y  = 0, f(0)
            right_x, right_y = image_width - 1, f(image_width - 1)

            if left_y < right_y:
                best_thetaC = math.atan2(abs(left_y - camera_y), (left_x - camera_x))

                if debug:
                    #draw the last possible line
                    cv2.line(img, (left_x, left_y), (camera_x, camera_y), (255, 128, 128), 2)
                    cv2.line(img, (left_x, left_y), (best_x1, best_y1), (255, 128, 128), 2)
            else:
                best_thetaC = math.atan2(abs(right_y - camera_y), (right_x - camera_x))

                if debug:
                    #draw the last possible line
                    cv2.line(img, (right_x, right_y), (camera_x, camera_y), (255, 128, 128), 2)
                    cv2.line(img, (right_x, right_y), (best_x1, best_y1), (255, 128, 128), 2)

            steering_angle = best_thetaC
        else:
            steering_angle = best_thetaB

        if (steering_angle - np.pi/2) * (last_steering_angle - np.pi/2) < 0:
            last = ImageProcessor._find_best_matched_line(None, last_steering_angle, tolerance, vectors)

            if last:
                last_distance, last_length, last_thetaA, last_thetaB, last_coord = last
                steering_angle = last_thetaB

                if debug:
                    #draw the last possible line
                    cv2.line(img, last_coord[:2], last_coord[2:], (255, 128, 128), 2)

        if debug:
            #draw the steering direction
            r = 60
            x = image_width / 2 + int(r * math.cos(steering_angle))
            y = image_height    - int(r * math.sin(steering_angle))
            cv2.line(img, (image_width / 2, image_height), (x, y), (255, 0, 255), 2)
            logit("line angle: %0.2f, steering angle: %0.2f, last steering angle: %0.2f" % (ImageProcessor.rad2deg(best_thetaA), ImageProcessor.rad2deg(np.pi/2-steering_angle), ImageProcessor.rad2deg(np.pi/2-last_steering_angle)))

        return (np.pi/2 - steering_angle)

    @staticmethod
    def find_track_angles(img_src, near, med, far, img_draw, color, debug = True):
        img = img_src.copy()

        image_height = img.shape[0]
        image_width  = img.shape[1]
        camera_x     = image_width / 2

        im2, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        area_dict = {}
        #依照面積排列
        for index in range(0,len(contours)):
            area = cv2.contourArea(contours[index])
            area_dict[index] = area

        #只取面積前兩大
        area_sort = sorted(area_dict.items(), key=lambda d:d[1], reverse=True)[0:2]

        point_all = []
        angle_all = []
        for item in area_sort:
            index = item[0]
            points = []
            angles = []
            #建立這個邊緣的mask
            mask = np.zeros(img.shape, dtype=np.uint8)
            #-1填滿
            cv2.drawContours(mask, contours, index, (255), -1)

            #取近一點
            steering_angle = np.nan
            _y, _x = np.where(mask[med:near,:] == 255)
            if len(_x)>0 and len(_y)>0:
                px = int(np.mean(_x))
                py = int(np.mean(_y) + med)
                points.append((px, py))
                steering_angle = math.atan2(image_height-py, (px - camera_x))
            angles.append(steering_angle)


            #取中間
            steering_angle = np.nan
            _y, _x = np.where(mask[far:med,:] == 255)
            if len(_x)>0 and len(_y)>0:
                px = int(np.mean(_x))
                py = int(np.mean(_y) + far)
                points.append((px, py))
                steering_angle = math.atan2(image_height-py, (px - camera_x))
            angles.append(steering_angle)

            #取遠方
            steering_angle = np.nan
            _y, _x = np.where(mask[0:far,:] == 255)
            if len(_x)>0 and len(_y)>0:
                px = int(np.mean(_x))
                py = int(np.mean(_y))
                points.append((px, py))
                steering_angle = math.atan2(image_height-py, (px - camera_x))
            angles.append(steering_angle)

            point_angle = np.nan
            if len(points) == 3:
                point_x = points[-1][0] - points[0][0]
                point_y = points[-1][1] - points[0][1]
                point_angle = math.atan2(point_y, point_x)
            #找不到三個點表示太彎
            else:
                point_angle = math.atan2(0,0)
            angles.append(point_angle)

            point_all.append(points)
            angle_all.append(angles)

            if debug:
                draw_heigh_offset_value = int(ConstValue.IMAGE_HEIGHT - image_height)
                draw_offset_array = np.array([0, draw_heigh_offset_value])
                contours[index] = contours[index] + draw_offset_array
                cv2.drawContours(img_draw, contours, index, color, 3)
                for point in points:
                    cv2.circle(img_draw, (point[0], point[1] + draw_heigh_offset_value), 3, color, 3, 8)
        
        return angle_all

    @staticmethod
    def find_track(img_road, img_wall_black, img_wall_yellow, img_draw, debug = True):
        draw_heigh_offset_value = int(ConstValue.IMAGE_HEIGHT * ConstValue.IMAGE_TRACK_HEIGHT)
        draw_offset_array = np.array([0, draw_heigh_offset_value])

        image_height = img_road.shape[0]
        image_width  = img_road.shape[1]
        camera_x     = image_width / 2

        #計算遠中近三點
        far = int(img_road.shape[0] * 0.2)
        med = int(img_road.shape[0] * 0.5)
        near = int(img_road.shape[0] * 0.7)

        angles_road = ImageProcessor.find_track_angles(img_road, near, med, far, img_draw, (255,0,0), debug)

        #所有牆
        img_wall = img_wall_black + img_wall_yellow

        wall_far = int(img_wall.shape[0] * 0.1)
        img_wall_bottom = img_wall[wall_far:,:]

        #計算牆數
        angles_wall = ImageProcessor.find_track_angles(img_wall_bottom, near, med, wall_far, img_draw, (255,255,255), debug)

        return angles_road, angles_wall

    #return 是否倒退, 角度
    @staticmethod
    def go_back(img_wall_black, img_wall_yellow, img_r, img_g, img_b, debug = True):
        # 38400 = 240*320*0.5
        __TOTAL_PIXEL = 38400
        # 20736 = __TOTAL_PIXEL * 0.7
        __THRESHOLD_PIXEL_MAX = 26880
        # 6144 = __TOTAL_PIXEL * 0.3
        __THRESHOLD_PIXEL_MIN = 11520

        #找到黑色牆
        _wb_y, _wb_x = np.where(img_wall_black==255)

        #找到黃色牆
        _wy_y, _wy_x = np.where(img_wall_yellow==255)

        #找到綠路
        _g_y, _g_x = np.where(img_g==255)

        #找到紅路
        LowerRed1 = np.array([0, 42, 46])
        _r_y, _r_x = np.where(img_r==255)

        #找到藍路
        _b_y, _b_x = np.where(img_b==255)

        _wb_p = len(_wb_y)
        _wy_p = len(_wy_y)
        _g_p = len(_g_y)
        _r_p = len(_r_y)
        _right_site_bg = _wb_p + _g_p
        _right_site_yr = _wy_p + _r_p
        _left_site_br = _wb_p + _r_p
        _left_site_yg = _wy_p + _g_p
        #print('_right_site_bg:{} _right_site_yr:{} _left_site_br:{} _left_site_yg:{}'.format(_right_site_bg, _right_site_yr, _left_site_br, _left_site_yg))
        #print('_wb_p:{} _wy_p:{} _g_p:{} _r_p:{}'.format(_wb_p, _wy_p, _g_p, _r_p))

        back_angle = 0
        is_goback = False
        #全部是牆壁
        if _wb_p >= __THRESHOLD_PIXEL_MAX or _wy_p >= __THRESHOLD_PIXEL_MAX:
            is_goback = True
            back_angle = 0
        #右邊牆壁：黑＋綠 or 黃＋紅
        #確保牆壁比道路多，且避免其中一種都是零
        elif _right_site_bg >= __THRESHOLD_PIXEL_MAX and (_wb_p > __THRESHOLD_PIXEL_MIN and _g_p > __THRESHOLD_PIXEL_MIN):
            is_goback = True
            back_angle = 40
        elif _right_site_yr >= __THRESHOLD_PIXEL_MAX and (_wy_p > __THRESHOLD_PIXEL_MIN and _r_p > __THRESHOLD_PIXEL_MIN):
            is_goback = True
            back_angle = 40
        #右邊牆壁：黑＋紅 or 黃＋綠
        #確保牆壁比道路多，且避免其中一種都是零
        elif _left_site_br >= __THRESHOLD_PIXEL_MAX and (_wb_p > __THRESHOLD_PIXEL_MIN and _r_p > __THRESHOLD_PIXEL_MIN):
            is_goback = True
            back_angle = -40
        elif _left_site_yg >= __THRESHOLD_PIXEL_MAX and (_wy_p > __THRESHOLD_PIXEL_MIN and _g_p > __THRESHOLD_PIXEL_MIN):
            is_goback = True
            back_angle = -40
        #print('{}:{}'.format(is_goback,back_angle))
        return is_goback, back_angle

    @staticmethod
    def is_narrow_road(img_r, img_g, img_b, debug = True):
        _r_y, _r_x = np.where(img_r==255)
        _g_y, _g_x = np.where(img_g==255)
        _b_y, _b_x = np.where(img_b==255)
        
        narrow_area = 200
        _p_r = len(_r_y)
        _p_g = len(_g_y)
        _p_b = len(_b_y)

        area_data = {'r': _p_r, 'g': _p_g, 'b': _p_b}
        area_sort = sorted(area_data.items(), key=lambda d:d[1], reverse=True)[0:3]
        road_list = []
        for item in area_sort:
            road_list.append(item[0])
        #print('r:{} g:{} b:{}'.format(len(_r_y), len(_g_y), len(_b_y)))
        if _p_r < narrow_area or _p_g < narrow_area or _p_b < narrow_area:
            return True, road_list
        return False, road_list

    @staticmethod
    def track_analyzer(img_src, img_draw, debug = True):
        bottom_half_ratios = (ConstValue.IMAGE_TRACK_HEIGHT, 1.0)
        bottom_half_slice  = slice(*(int(x * img_src.shape[0]) for x in bottom_half_ratios))
        bottom_half        = img_src[bottom_half_slice, :, :]

        img_proc = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2HSV)

        image_height = img_proc.shape[0]
        image_width  = img_proc.shape[1]
        camera_x     = image_width / 2

        struct_element = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5));

        #找到黑色牆
        LowerBlack = np.array([0, 0, 0])
        UpperBlack = np.array([180, 255, 46])
        img_wall_black = cv2.inRange(img_proc, LowerBlack, UpperBlack)
        #先縮白色再擴張白色，去雜訊
        img_wall_black = cv2.erode(img_wall_black,struct_element, iterations=1)
        img_wall_black = cv2.dilate(img_wall_black,struct_element, iterations=1)

        #找到黃色牆
        LowerYellow = np.array([26, 43, 46])
        UpperYellow = np.array([34, 255, 255])
        img_wall_yellow = cv2.inRange(img_proc, LowerYellow, UpperYellow)
        #先縮白色再擴張白色，去雜訊
        img_wall_yellow = cv2.erode(img_wall_yellow,struct_element, iterations=1)
        img_wall_yellow = cv2.dilate(img_wall_yellow,struct_element, iterations=1)

        #找到綠路
        LowerGreen = np.array([35, 42, 46])
        UpperGreen = np.array([77, 255, 255])
        img_g = cv2.inRange(img_proc, LowerGreen, UpperGreen)
        #先縮白色再擴張白色，去雜訊
        img_g = cv2.erode(img_g,struct_element, iterations=1)
        img_g = cv2.dilate(img_g,struct_element, iterations=1)

        #找到紅路
        LowerRed1 = np.array([0, 42, 46])
        UpperRed1 = np.array([10, 255, 255])
        img_r1 = cv2.inRange(img_proc, LowerRed1, UpperRed1)
        LowerRed2 = np.array([156, 42, 46])
        UpperRed2 = np.array([180, 255, 255])
        img_r2 = cv2.inRange(img_proc, LowerRed2, UpperRed2)
        img_r = img_r1 | img_r2        
        #先縮白色再擴張白色，去雜訊
        img_r = cv2.erode(img_r,struct_element, iterations=1)
        img_r = cv2.dilate(img_r,struct_element, iterations=1)

        #找到藍路
        LowerBlue = np.array([100, 42, 46])
        UpperBlue = np.array([124, 255, 255])
        img_b = cv2.inRange(img_proc, LowerBlue, UpperBlue)
        #先縮白色再擴張白色，去雜訊
        img_b = cv2.erode(img_b,struct_element, iterations=1)
        img_b = cv2.dilate(img_b,struct_element, iterations=1)

        is_goback, back_angle = ImageProcessor.go_back(img_wall_black, img_wall_yellow, img_r, img_g, img_b, debug)

        angles_road, angles_wall = ImageProcessor.find_track(img_b, img_wall_black, img_wall_yellow, img_draw, debug)

        is_narrow, road_list = ImageProcessor.is_narrow_road(img_r, img_g, img_b)

        return angles_road, angles_wall, is_goback, back_angle, is_narrow, road_list

    @staticmethod
    def find_steering_angle_by_color(img, last_steering_angle, debug = True):
        r, g, b      = cv2.split(img)
        image_height = img.shape[0]
        image_width  = img.shape[1]
        camera_x     = image_width / 2
        image_sample = slice(0, int(image_height * 0.2))
        sr, sg, sb   = r[image_sample, :], g[image_sample, :], b[image_sample, :]
        track_list   = [sr, sg, sb]
        tracks       = map(lambda x: len(x[x > 20]), [sr, sg, sb])
        tracks_seen  = filter(lambda y: y > 50, tracks)

        if len(tracks_seen) == 0:
            return 0.0

        maximum_color_idx = np.argmax(tracks, axis=None)
        _target = track_list[maximum_color_idx]
        _y, _x = np.where(_target == 255)
        px = np.mean(_x)
        py = np.mean(_y)
        cv2.circle(img, (int(px),int(py)),20,(255,255,0),3,8)
        steering_angle = math.atan2(image_height, (px - camera_x))

        if debug:
            #draw the steering direction
            r = 60
            x = image_width / 2 + int(r * math.cos(steering_angle))
            y = image_height    - int(r * math.sin(steering_angle))
            cv2.line(img, (image_width / 2, image_height), (x, y), (255, 0, 255), 2)
            logit("steering angle: %0.2f, last steering angle: %0.2f" % (ImageProcessor.rad2deg(steering_angle), ImageProcessor.rad2deg(np.pi/2-last_steering_angle)))

        return (np.pi/2 - steering_angle) * 2.0


class AutoDrive(object):
    STEERING_PID_Kp             = 0.3
    STEERING_PID_Ki             = 0.01
    STEERING_PID_Kd             = 0.1
    STEERING_PID_max_integral   = 10
    THROTTLE_PID_Kp             = 0.02
    THROTTLE_PID_Ki             = 0.005
    THROTTLE_PID_Kd             = 0.02
    THROTTLE_PID_max_integral   = 0.5
    MAX_STEERING_HISTORY        = 3
    MAX_THROTTLE_HISTORY        = 3
    DEFAULT_SPEED               = 0.5

    debug = True

    def __init__(self, car, model_path, record_folder = None):
        self._record_folder    = record_folder

        self._steering_pid     = PID(Kp=self.STEERING_PID_Kp  , Ki=self.STEERING_PID_Ki  , Kd=self.STEERING_PID_Kd  , max_integral=self.STEERING_PID_max_integral  )
        self._throttle_pid     = PID(Kp=self.THROTTLE_PID_Kp  , Ki=self.THROTTLE_PID_Ki  , Kd=self.THROTTLE_PID_Kd  , max_integral=self.THROTTLE_PID_max_integral  )
        self._throttle_pid.assign_set_point(self.DEFAULT_SPEED)
        self._steering_history = []
        self._throttle_history = []
        self._car = car
        self._car.register(self)

        self.debug = ConstValue.DEBUG

        self.sign = Sign()

        self._speedup = collections.deque(maxlen=10)

        self._turn_degree = np.nan
        self._turn_angle_left = math.atan2(1,-1)
        self._turn_angle_right = math.atan2(1,1)

        self._last_current_angle = np.nan

        self.stuck_counter = 0

        self.model = ks_load_model(model_path)
        self.is_narrow = False

        self.pid = False


    def pid_bot(self, angles_road, angles_wall, speed, isFindSign, direction, speed_rate):
        #speed = 1
        #牆壁角度
        wall_angle = np.nan
        #轉向角度
        turn_angle = np.nan

        if len(angles_wall) > 0:
            #找最近的牆壁角度
            #從遠到近，找到一個即可
            if not np.isnan(angles_wall[0][2]):
                wall_angle = angles_wall[0][2]
            elif not np.isnan(angles_wall[0][1]):
                wall_angle = angles_wall[0][1]
            elif not np.isnan(angles_wall[0][0]):
                wall_angle = angles_wall[0][0]

            wall_degree = math.degrees(wall_angle)

            #檢查轉彎校正
            if self._turn_degree == self._turn_angle_left:
                speed_rate = 0.6
                #如果號誌向左，且牆壁在左邊，就取消轉彎校正
                # 90~135
                if (wall_degree > 90) and (wall_degree <= 140):
                    self._turn_degree = np.nan
            elif self._turn_degree == self._turn_angle_right:
                speed_rate = 0.6
                # 45~90
                #如果號誌向左，且牆壁在左邊，就取消轉彎校正
                if wall_degree < 90 and wall_degree>=40:
                    self._turn_degree = np.nan
        #找到號誌
        if isFindSign == True:
            wall_degree = math.degrees(wall_angle)
            #號誌向左
            if direction == Sign.DIRECTION_LEFT:
                #如果牆壁不在左邊，採取動作
                if not ((wall_degree > 90) and (wall_degree <= 180)):
                    self._turn_degree = self._turn_angle_left
            #號誌向右
            elif direction == Sign.DIRECTION_RIGHT:
                #如果牆壁不在右邊，採取動作
                if not (wall_degree < 90):
                    self._turn_degree = self._turn_angle_right

        current_angle = np.nan

        if len(angles_road) > 0:
            angle_near = angles_road[0][0]
            angle_med = angles_road[0][1]
            angle_far = angles_road[0][2]
            logit('near: {}, med: {}, far: {}'.format(angle_near, angle_med, angle_far))

            if np.isnan(angle_med):
                angle_med = angle_near
            if np.isnan(angle_far):
                angle_far = angle_med
            if not np.isnan(self._turn_degree):
                #angle_far = angle_far + self._turn_degree
                angle_far = self._turn_degree
            current_angle = np.pi - angle_far * 2.0

            point_angle = math.fabs(angles_road[0][3])
            point_angle = math.fabs(math.degrees(point_angle)-90.0)
            logit('track_angle: {}'.format(point_angle))
            
            far_from_track = math.fabs(angle_near*2 - np.pi)

            if not np.isnan(self._turn_degree):
                far_from_track = 0
                point_angle = 90
            if far_from_track > 1.8:
                if point_angle>70:
                    current_angle = np.pi - angle_med * 2.0
                    logit('far from track, big turn, choose medium point')
                else:
                    logit('far from track, get back to track faster')
                    current_angle = current_angle * 1.5
                speed_rate = 0.8

            speed_out = 1
            if point_angle > 70:
                if len(self._speedup)==0:
                    for i in range(0,10):
                        self._speedup.append(1.1**i)
                elif len(self._speedup) > 0:
                    speed_out = self._speedup.popleft()
            else:
                self._speedup.clear()

            if point_angle >= 89.5:
                current_angle = current_angle/2.2
                self._throttle_pid.assign_set_point(0.3*speed_rate*speed_out)
            elif point_angle > 86:
                current_angle = current_angle/2.5
                self._throttle_pid.assign_set_point(0.6*speed_rate*speed_out)
            elif point_angle > 80.0:
                current_angle = current_angle/3
                self._throttle_pid.assign_set_point(0.8*speed_rate*speed_out)
            elif point_angle > 70.0:
                current_angle = current_angle/3.3
                self._throttle_pid.assign_set_point(1.2*speed_rate*speed_out)
            elif point_angle > 60.0:
                current_angle = current_angle/3.9
                self._throttle_pid.assign_set_point(2.5*speed_rate)
            elif point_angle > 45.0:
                current_angle = current_angle/4.3
                self._throttle_pid.assign_set_point(2.5*speed_rate)
            elif point_angle > 30.0:
                current_angle = current_angle/4.5
                self._throttle_pid.assign_set_point(2.8*speed_rate)
            elif point_angle > 10.0:
                current_angle = current_angle/5
                self._throttle_pid.assign_set_point(3*speed_rate)
            else:
                current_angle = current_angle/5
                self._throttle_pid.assign_set_point(3.5*speed_rate)

        return current_angle, speed


    def detect_stuck(self, speed):
        if speed < ConstValue.MIN_SPEED:
            self.stuck_counter += 1
        else:
            self.stuck_counter -= 1
            if self.stuck_counter < 0:
                self.stuck_counter = 0
        if self.stuck_counter > ConstValue.STUCK_COUNTER:
            return True
        else:
            return False

    def on_dashboard(self, src_img, last_steering_angle, speed, throttle, info):
        draw_img = src_img.copy()
        top_img       = ImageProcessor.preprocessTop(src_img.copy())
        _y, _x = np.where(top_img==0)

        speed_rate = 1
        isFindSign = False
        direction = None

        if len(_y)>50:
            speed_rate = 0.85
            # search sign
            # saving CPU
            isFindSign, direction = self.sign.find_sign(top_img, draw_img)


        angles_road, angles_wall, is_goback, back_angle, is_narrow, road_list = ImageProcessor.track_analyzer(src_img, draw_img, self.debug)

        is_stuck = self.detect_stuck(speed)

        #確保pid跑在窄路中
        if self.is_narrow == True and self.pid == True:
            if is_narrow == False:
                self.is_narrow = False
                self.pid = False

        #確保跑在窄路中
        if is_narrow == True and np.isnan(self._turn_degree):
            self.is_narrow = True

        #找到號誌，切換成pid
        if isFindSign == True:
            self.pid = True

        if self.pid == True:
            pid_current_angle, pid_speed = self.pid_bot(angles_road, angles_wall, speed, isFindSign, direction, speed_rate)
            if np.isnan(pid_current_angle):
                is_stuck = True
            else:
                steering_angle = self._steering_pid.update(-pid_current_angle)
                throttle       = self._throttle_pid.update(pid_speed)

                #smooth the control signals
                self._steering_history.append(steering_angle)
                self._steering_history = self._steering_history[-self.MAX_STEERING_HISTORY:]
                self._throttle_history.append(throttle)
                self._throttle_history = self._throttle_history[-self.MAX_THROTTLE_HISTORY:]

                send_steering_angle = sum(self._steering_history)/self.MAX_STEERING_HISTORY
                send_steering_angle = ImageProcessor.rad2deg(send_steering_angle)
                send_throttle = sum(self._throttle_history)/self.MAX_THROTTLE_HISTORY

                self._last_current_angle = send_steering_angle

        else:
            image_original = utils.preprocess(src_img)  # apply the preprocessing
            image = np.array([image_original])  # the model expects 4D array
            # speed = np.array([speed])  # the model expects 4D array

            # predict the steering angle for the image
            parameter = self.model.predict(image, batch_size=1)
            send_steering_angle, send_throttle = float(parameter[0][0]), float(parameter[0][1])
            
            if abs(send_steering_angle) >= 20:
                send_throttle = 0.1
            elif abs(send_steering_angle) >= 10:
                send_throttle = 0.09

        if is_goback:
            if is_goback:
                self._car.control(back_angle, -0.5)
                return
            else:
                send_steering_angle = self._last_current_angle
                send_throttle = -send_throttle

        #卡住
        if is_stuck:
            if 'b' == road_list[0]:
                if 'r' == road_list[1]:
                    send_steering_angle = -40
                else:
                    send_steering_angle = 40
            elif 'r' == road_list[0]:
                send_steering_angle = -40
            else:
                send_steering_angle = 40

            send_throttle = -5

        if self.debug:
            ImageProcessor.show_image(draw_img, "source")
            #ImageProcessor.show_image(track_img, "track")
            ImageProcessor.show_image(top_img, "top - sign")
            # logit("steering PID: %0.2f (%0.2f) => %0.2f (%0.2f)" % (current_angle, ImageProcessor.rad2deg(current_angle), steering_angle, ImageProcessor.rad2deg(steering_angle)))
            # logit("throttle PID: %0.4f => %0.4f" % (speed, throttle))
            # logit("info: %s" % repr(info))

        if self._record_folder:
            suffix = "-deg%0.3f" % (ImageProcessor.rad2deg(steering_angle))
            ImageProcessor.save_image(self._record_folder, src_img  , prefix = "cam", suffix = suffix)
            #ImageProcessor.save_image(self._record_folder, track_img, prefix = "trk", suffix = suffix)
        
        #error handling
        if np.isnan(send_steering_angle):
            send_steering_angle = 0
        self._car.control(send_steering_angle, send_throttle)


class Car(object):
    MAX_STEERING_ANGLE = 40.0


    def __init__(self, control_function):
        self._driver = None
        self._control_function = control_function


    def register(self, driver):
        self._driver = driver


    def on_dashboard(self, dashboard):
        #normalize the units of all parameters
        last_steering_angle = np.pi/2 - float(dashboard["steering_angle"]) / 180.0 * np.pi
        throttle            = float(dashboard["throttle"])
        brake               = float(dashboard["brakes"])
        speed               = float(dashboard["speed"])
        img                 = ImageProcessor.bgr2rgb(np.asarray(Image.open(BytesIO(base64.b64decode(dashboard["image"])))))
        del dashboard["image"]
        logit('{} {}'.format(datetime.now(), dashboard))
        total_time = float(dashboard["time"])
        elapsed    = total_time

        if elapsed >600:
            logit("elapsed: " +str(elapsed))
            send_restart()

        info = {
            "lap"    : int(dashboard["lap"]) if "lap" in dashboard else 0,
            "elapsed": elapsed,
            "status" : int(dashboard["status"]) if "status" in dashboard else 0,
        }
        self._driver.on_dashboard(img, last_steering_angle, speed, throttle, info)


    def control(self, steering_angle, throttle):
        #convert the values with proper units
        steering_angle = min(max(steering_angle, -Car.MAX_STEERING_ANGLE), Car.MAX_STEERING_ANGLE)
        self._control_function(steering_angle, throttle)


if __name__ == "__main__":
    import shutil
    import argparse
    from datetime import datetime

    import socketio
    import eventlet
    import eventlet.wsgi
    from flask import Flask

    parser = argparse.ArgumentParser(description='AutoDriveBot')
    parser.add_argument("-r", 
        "--record",
        type=str,
        default='',
        metavar='images',
        help='Path to image folder to record the images.'
    )
    parser.add_argument("-nd", 
        "--no_debug",
        required=False,
        action='store_true',
        help='Stop debug mode, hide images and logs.'
    )
    parser.add_argument("--model",
        required=False,
        type=str,
        default='model.h5',
        metavar='model.h5',
        help='Load the CNN model.'
    )    

    args = parser.parse_args()

    date_path = datetime.now().strftime('%Y%m%d-%H%M%S')

    if args.record:
        args.record = os.path.join(date_path, args.record)
        if not os.path.exists(args.record):
            os.makedirs(args.record)
        logit("Start recording images to %s..." % args.record)
    if args.no_debug:
        ConstValue.DEBUG=False
    else:
        logger.addHandler(ch)

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

    car = Car(control_function = send_control)
    drive = AutoDrive(car, args.model, args.record)

    @sio.on('telemetry')
    def telemetry(sid, dashboard):
        if dashboard:
            car.on_dashboard(dashboard)
        else:
            sio.emit('manual', data={}, skip_sid=True)

    @sio.on('connect')
    def connect(sid, environ):
        car.control(0, 0)

    app = socketio.Middleware(sio, Flask(__name__))
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

    cv2.destroyAllWindows()

# vim: set sw=4 ts=4 et :

