#!/usr/bin/env python
# -*- coding: utf-8 -*-
import base64
import math

import cv2
import numpy as np
from keras.models import model_from_json

from behavior_cloning.train import img_pre_processing
from interface.car import Car
from traffic_sign.TrafficSign import identifyTrafficSign
from .image_processor import ImageProcessor


class BeCar(Car):
    debug = False

    def __init__(self, model_path, do_sign_detection=True):
        self.model = self.load_model(model_path)
        self.last_steering_angle = 0

        # Handle sign detection
        self.do_sign_detection = do_sign_detection
        self.last_detected_sign = None
        self.acting_new_sign = None
        self.accumulated_acting_frames = 0
        self.MAX_ACCUMULATED_ACTING_FRAMES = 5
        if self.do_sign_detection:
            self._sign = identifyTrafficSign()
        else:
            self._sign = None

        self._turn_degree = np.nan
        self._turn_angle_left = -7.5
        self._turn_angle_right = 10.5

    @staticmethod
    def load_model(path):
        with open(path, 'r') as jfile:
            json_model = jfile.read()
            model = model_from_json(json_model)

        model.compile("adam", "mse")
        weights_file = path.replace('json', 'h5')
        model.load_weights(weights_file)

        return model

    def logit(self, *args, sep=' ', end='\n'):
        if self.debug:
            print(*args, sep=sep, end=end)

    def print_layers(self):
        for layer in self.model.layers:
            self.logit(layer.name, layer.input, layer.output)
            self.logit(layer.input_shape, layer.output_shape)
            self.logit()

    def make_wall_to_same_color(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([21, 39, 64])
        upper_yellow = np.array([40, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        frame[yellow_mask > 0] = (0, 0, 0)
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
        img_area = len(sr) * len(sr[0])
        _y_w, _x_w = np.where(sw == 0)

        if len(_x_w) > img_area * 0.8:
            if move_forward is True:
                return last_steering_angle * 2.5
            else:
                return -last_steering_angle * 2.5
        return None

    def get_detected_sign(self, src_img):
        # TODO: Navigate to correct track based on detected traffic sign
        traffic_sign = self._sign.detect(src_img)
        traffic_sign = None if traffic_sign == "None" else traffic_sign
        if traffic_sign:
            if self.last_detected_sign != traffic_sign:  # We detected a new sign
                self.logit("New sign detected:", traffic_sign)
                self.accumulated_acting_frames = 0
                self.acting_new_sign = traffic_sign
                self.last_detected_sign = traffic_sign

        if self.acting_new_sign:
            self.logit("Responding to", self.acting_new_sign)

            self.accumulated_acting_frames += 1
            if self.accumulated_acting_frames >= self.MAX_ACCUMULATED_ACTING_FRAMES:
                # TODO: Should consider car speed for response time
                self.accumulated_acting_frames = 0
                self.acting_new_sign = None
                self.last_detected_sign = None

        return self.acting_new_sign

    # return 是否倒退, 角度
    @staticmethod
    def go_back(img_src, debug=False):
        # 30720 = 240*0.4*320
        __TOTAL_PIXEL = 30720
        # 20736 = __TOTAL_PIXEL * 0.8
        __THRESHOLD_PIXEL_MAX = 24576
        # 6144 = __TOTAL_PIXEL * 0.2
        __THRESHOLD_PIXEL_MIN = 3072

        wall_ratios = (0.6, 1)
        wall_slice = slice(*(int(x * img_src.shape[0]) for x in wall_ratios))
        wall_img = img_src[wall_slice, :, :]

        struct_element = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5));

        img_proc = cv2.cvtColor(wall_img, cv2.COLOR_BGR2HSV)

        # 找到黑色牆
        LowerBlack = np.array([0, 0, 0])
        UpperBlack = np.array([180, 255, 46])
        img_wall_black = cv2.inRange(img_proc, LowerBlack, UpperBlack)
        # 先縮白色再擴張白色，去雜訊
        img_wall_black = cv2.erode(img_wall_black, struct_element, iterations=1)
        img_wall_black = cv2.dilate(img_wall_black, struct_element, iterations=1)
        _wb_y, _wb_x = np.where(img_wall_black == 255)

        # 找到黃色牆
        LowerYellow = np.array([26, 43, 46])
        UpperYellow = np.array([34, 255, 255])
        img_wall_yellow = cv2.inRange(img_proc, LowerYellow, UpperYellow)
        # 先縮白色再擴張白色，去雜訊
        img_wall_yellow = cv2.erode(img_wall_yellow, struct_element, iterations=1)
        img_wall_yellow = cv2.dilate(img_wall_yellow, struct_element, iterations=1)
        _wy_y, _wy_x = np.where(img_wall_yellow == 255)

        # 找到綠路
        LowerGreen = np.array([35, 42, 46])
        UpperGreen = np.array([77, 255, 255])
        img_g = cv2.inRange(img_proc, LowerGreen, UpperGreen)
        # 先縮白色再擴張白色，去雜訊
        img_g = cv2.erode(img_g, struct_element, iterations=1)
        img_g = cv2.dilate(img_g, struct_element, iterations=1)
        _g_y, _g_x = np.where(img_g == 255)
        # 找到紅路
        LowerRed1 = np.array([0, 42, 46])
        UpperRed1 = np.array([10, 255, 255])
        img_r1 = cv2.inRange(img_proc, LowerRed1, UpperRed1)
        LowerRed2 = np.array([156, 42, 46])
        UpperRed2 = np.array([180, 255, 255])
        img_r2 = cv2.inRange(img_proc, LowerRed2, UpperRed2)
        img_r = img_r1 | img_r2
        # 先縮白色再擴張白色，去雜訊
        img_r = cv2.erode(img_r, struct_element, iterations=1)
        img_r = cv2.dilate(img_r, struct_element, iterations=1)
        _r_y, _r_x = np.where(img_r == 255)

        _wb_p = len(_wb_y)
        _wy_p = len(_wy_y)
        _g_p = len(_g_y)
        _r_p = len(_r_y)
        _right_site_bg = _wb_p + _g_p
        _right_site_yr = _wy_p + _r_p
        _left_site_br = _wb_p + _r_p
        _left_site_yg = _wy_p + _g_p

        back_angle = 0
        is_goback = False
        # 全部是牆壁
        if _wb_p >= __THRESHOLD_PIXEL_MAX or _wy_p >= __THRESHOLD_PIXEL_MAX:
            is_goback = True
            back_angle = 0
        # 右邊牆壁：黑＋綠 or 黃＋紅
        # 確保牆壁比道路多，且避免其中一種都是零
        elif _right_site_bg >= __THRESHOLD_PIXEL_MAX and (
                        _wb_p > __THRESHOLD_PIXEL_MIN and _g_p > __THRESHOLD_PIXEL_MIN):
            is_goback = True
            back_angle = 40
        elif _right_site_yr >= __THRESHOLD_PIXEL_MAX and (
                        _wy_p > __THRESHOLD_PIXEL_MIN and _r_p > __THRESHOLD_PIXEL_MIN):
            is_goback = True
            back_angle = 40
        # 右邊牆壁：黑＋紅 or 黃＋綠
        # 確保牆壁比道路多，且避免其中一種都是零
        elif _left_site_br >= __THRESHOLD_PIXEL_MAX and (
                        _wb_p > __THRESHOLD_PIXEL_MIN and _r_p > __THRESHOLD_PIXEL_MIN):
            is_goback = True
            back_angle = -40
        elif _left_site_yg >= __THRESHOLD_PIXEL_MAX and (
                        _wy_p > __THRESHOLD_PIXEL_MIN and _g_p > __THRESHOLD_PIXEL_MIN):
            is_goback = True
            back_angle = -40

        return is_goback, back_angle

    def find_track_angles(self, img_src, near, med, far):
        img = img_src.copy()

        image_height = img.shape[0]
        image_width = img.shape[1]
        camera_x = image_width / 2

        im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        area_dict = {}
        # 依照面積排列
        for index in range(0, len(contours)):
            area = cv2.contourArea(contours[index])
            area_dict[index] = area

        area_sort = sorted(area_dict.items(), key=lambda d: d[1], reverse=True)[0:2]
        # 只取面積前兩大
        point_all = []
        angle_all = []
        for item in area_sort:
            index = item[0]
            points = []
            angles = []
            # 建立這個邊緣的mask
            mask = np.zeros(img.shape, dtype=np.uint8)
            # -1填滿
            cv2.drawContours(mask, contours, index, (255), -1)

            # 取近一點
            steering_angle = np.nan
            _y, _x = np.where(mask[med:near, :] == 255)
            if len(_x) > 0 and len(_y) > 0:
                px = int(np.mean(_x))
                py = int(np.mean(_y) + med)
                points.append((px, py))
                steering_angle = math.atan2(image_height - py, (px - camera_x))
            angles.append(steering_angle)

            # 取中間
            steering_angle = np.nan
            _y, _x = np.where(mask[far:med, :] == 255)
            if len(_x) > 0 and len(_y) > 0:
                px = int(np.mean(_x))
                py = int(np.mean(_y) + far)
                points.append((px, py))
                steering_angle = math.atan2(image_height - py, (px - camera_x))
            angles.append(steering_angle)

            # 取遠方
            steering_angle = np.nan
            _y, _x = np.where(mask[0:far, :] == 255)
            if len(_x) > 0 and len(_y) > 0:
                px = int(np.mean(_x))
                py = int(np.mean(_y))
                points.append((px, py))
                steering_angle = math.atan2(image_height - py, (px - camera_x))
            angles.append(steering_angle)

            point_angle = np.nan
            if len(points) == 3:
                point_x = points[-1][0] - points[0][0]
                point_y = points[-1][1] - points[0][1]
                point_angle = math.atan2(point_y, point_x)
            # 找不到三個點表示太彎
            else:
                point_angle = math.atan2(0, 0)
            angles.append(point_angle)

            point_all.append(points)
            angle_all.append(angles)

        return angle_all

    # return wall_angle[near-far][angle:near-med-far]
    def find_wall(self, img_src):
        bottom_half_ratios = (0.5, 1.0)
        bottom_half_slice = slice(*(int(x * img_src.shape[0]) for x in bottom_half_ratios))
        bottom_half = img_src[bottom_half_slice, :, :]

        img_proc = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2HSV)
        struct_element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3));

        image_height = img_proc.shape[0]
        image_width = img_proc.shape[1]
        camera_x = image_width / 2

        far = int(img_proc.shape[0] * 0.2)
        med = int(img_proc.shape[0] * 0.5)
        near = int(img_proc.shape[0] * 0.7)

        # 找到黑色牆
        LowerBlack = np.array([0, 0, 0])
        UpperBlack = np.array([180, 255, 46])
        img_wall_black = cv2.inRange(img_proc, LowerBlack, UpperBlack)

        # 找到黃色牆
        LowerYellow = np.array([26, 43, 46])
        UpperYellow = np.array([34, 255, 255])
        img_wall_yellow = cv2.inRange(img_proc, LowerYellow, UpperYellow)

        # 所有牆
        img_wall = img_wall_black + img_wall_yellow
        # 侵蝕白色，去掉雜訊
        img_wall = cv2.erode(img_wall, struct_element)

        wall_far = int(img_proc.shape[0] * 0.1)
        img_wall_bottom = img_wall[wall_far:, :]

        # 計算牆數
        angles_wall = self.find_track_angles(img_wall_bottom, near, med, wall_far)

        return angles_wall

    def on_dashboard(self, data):
        # The current steering angle of the car
        # steering_angle = data["steering_angle"]
        # The current throttle of the car
        # throttle = data["throttle"]
        # The current speed of the car
        # speed = data["speed"]
        # The current image from the center camera of the car
        image_rgb = cv2.imdecode(np.fromstring(base64.b64decode(data["image"]), np.uint8), flags=cv2.IMREAD_COLOR)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        is_goback, back_angle = self.go_back(image_rgb)
        if is_goback:
            # Handle car crash
            new_steering_angle, new_throttle = back_angle, -1
        else:
            detected_sign = None
            if self.do_sign_detection:
                detected_sign = self.get_detected_sign(image_rgb)

            img = img_pre_processing(image_bgr)
            img_batch = img[None, :, :, :].astype('float')

            prediction = self.model.predict(img_batch, batch_size=1)

            new_steering_angle = prediction[0][0]
            new_throttle = prediction[0][1] - prediction[0][2]  # throttle - brake

            angles_wall = self.find_wall(image_rgb)
            # 牆壁角度
            wall_angle = np.nan
            # 轉向角度
            self._turn_degree = np.nan
            if len(angles_wall) > 0:
                # 找最近的牆壁角度
                # 從遠到近，找到一個即可
                if not np.isnan(angles_wall[0][2]):
                    wall_angle = angles_wall[0][2]
                elif not np.isnan(angles_wall[0][1]):
                    wall_angle = angles_wall[0][1]
                elif not np.isnan(angles_wall[0][0]):
                    wall_angle = angles_wall[0][0]

                wall_degree = math.degrees(wall_angle)

                # 檢查轉彎校正
                if self._turn_degree == self._turn_angle_left:
                    # 降速讓轉彎更順暢
                    # 如果號誌向左，且牆壁在左邊，就取消轉彎校正
                    if (wall_degree > 90) and (wall_degree <= 180):
                        self._turn_degree = np.nan
                elif self._turn_degree == self._turn_angle_right:
                    # 降速讓轉彎更順暢
                    # 如果號誌向左，且牆壁在左邊，就取消轉彎校正
                    if wall_degree < 90:
                        self._turn_degree = np.nan

            # TODO: Apply proper angles when sign is detected
            if detected_sign in ('ForkLeft', 'ForkRight'):
                self.logit("Turn", detected_sign)

                wall_degree = math.degrees(wall_angle)

                if detected_sign == 'ForkLeft':
                    # 如果牆壁不在左邊，採取動作
                    if not ((wall_degree > 90) and (wall_degree <= 180)):
                        self._turn_degree = self._turn_angle_left
                        # new_steering_angle += -7.5
                else:
                    # 如果牆壁不在右邊，採取動作
                    # if not (wall_degree < 90):
                    self._turn_degree = self._turn_angle_right
                        # new_steering_angle += 7.5
                new_throttle = 0.01

            if not np.isnan(self._turn_degree):
                new_steering_angle += self._turn_degree
                self.logit('turn: {}'.format(new_steering_angle))

            self.last_steering_angle = new_steering_angle

        return new_steering_angle, new_throttle
