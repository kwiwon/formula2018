# encoding=utf-8
#!env python
#
# Q-Team self-driving bot
#
# Revision:      v1.0
# Released Date: Sep 26, 2018
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

# ['ForkLeft', 'ForkRight', 'TurnLeft', 'TurnRight', 'UTurnLeft', 'UTurnRight']
TRACK_1 = ['TurnLeft', 'TurnLeft', 'TurnLeft', 'ForkRight', 'TurnLeft', 'UTurnRight', 'TurnLeft', 'TurnLeft', 'TurnRight']
TRACK_4 = ['TurnLeft', 'ForkRight', 'TurnLeft', 'TurnLeft', 'ForkLeft', 'TurnLeft']
WEEK_2  = ['TurnLeft', 'ForkRight', 'ForkRight', 'TurnLeft', 'ForkLeft', 'ForkLeft']
TRAFFIC_SIGN_ORDER = WEEK_2

FLAG_SHOW_WINDOW = False

class twQTeamPID(object):

    def __init__(self, mode, Kp, Ki, Kd, max_integral, min_interval = 0.001, set_point = 0.0, last_time = None):

        self.m_Kp           = Kp
        self.m_Ki           = Ki
        self.m_Kd           = Kd
        self.m_ori_Kp       = Kp
        self.m_ori_Ki       = Ki
        self.m_ori_Kd       = Kd
        self.m_min_interval = min_interval
        self.m_max_integral = max_integral
        self.m_mode         = mode

        self.m_set_point    = set_point
        self.m_last_time    = last_time if last_time is not None else time()
        self.m_p_value      = 0.0
        self.m_i_value      = 0.0
        self.m_d_value      = 0.0
        self.m_delta_time   = 0.0
        self.m_delta_error  = 0.0
        self.m_last_error   = 0.0
        self.m_output       = 0.0

        self.m_steering_variant = []
        self.m_stability    = True
        self.m_kp_decrease  = 0.01
        self.m_ki_decrease  = 0.0001
        self.m_kd_decrease  = 0.001
        self.m_brakes       = 0.01
        self.m_velocity     = 0.01
        self.m_cur_increase = -1
        self.m_increase     = 0.01


    def stability_of_wheel(self,wheel_variants):
        variant_diff=1
        variant_count=0
        current_site=-1
        # for i in range (len(wheel_variants)):
        #     if math.fabs(wheel_variants[i]) >= 0.01:
        #         if current_site==-1:
        #             current_site=i
        #             variant_diff = wheel_variants[i]
        #         else:
        #             if math.fabs(current_site-i)==1:
        #                 if wheel_variants[i] > 0:
        #                     if variant_diff < 0:
        #                         variant_count += 1
        #                     variant_diff = wheel_variants[i]
        #                 else:
        #                     if variant_diff > 0:
        #                         variant_count += 1
        #                     variant_diff = wheel_variants[i]
        #                 current_site=i
        #             else:
        #                 current_site=i
        #                 variant_diff = wheel_variants[i]
        # stability = variant_count / (len(wheel_variants) * 1.0)
        # return stability
        for variant in wheel_variants:
            if math.fabs(variant)>=0.01:
                if variant_diff==1:
                    variant_diff=variant
                if variant>0:
                    if variant_diff<0:
                        variant_count+=1
                    variant_diff = variant
                else:
                    if variant_diff>0:
                        variant_count+=1
                    variant_diff = variant
        stability=variant_count / (len(wheel_variants) * 1.0)
        return stability

    def stability_of_wheel_2(self,wheel_variants):

        keep_stable = 0
        nCount      = 0
        stability   = 0

        for cnt in range(0, len(wheel_variants)):
            if math.fabs(wheel_variants[cnt]) >= 0.05:
                keep_stable += math.fabs(wheel_variants[cnt])
                nCount += 1

        if nCount != 0:
            stability = keep_stable / nCount

        return stability

    def update_wheel(self, angle_value, error, delta_time, delta_error, cur_time):

        self.m_steering_variant.append(angle_value)
        self.m_p_value = error
        self.m_i_value = min(max(error * delta_time, -self.m_max_integral), self.m_max_integral)
        self.m_d_value = delta_error / delta_time if delta_time > 0 else 0.0

        # update wheel
        # if len(self.m_steering_variant) >= 3:
        #     probability = self.stability_of_wheel_2(self.m_steering_variant[(len(self.m_steering_variant) - 3):len(self.m_steering_variant)])
        #     if probability >= 0.1 and probability < 0.7:
        #         if self.m_Kp > 0.06:
        #             self.m_Kp -= self.m_kp_decrease
        #         if self.m_Ki > 0.003:
        #             self.m_Ki -= self.m_ki_decrease
        #         if self.m_Kd > 0.03:
        #             self.m_Kd -= self.m_kd_decrease
        #         self.stability = False
        #     elif probability >= 0.7:
        #         if self.m_Kp > 0.18:
        #             self.m_Kp -= self.m_kp_decrease
        #         if self.m_Ki > 0.008:
        #             self.m_Ki -= self.m_ki_decrease
        #         if self.m_Kd > 0.08:
        #             self.m_Kd -= self.m_kd_decrease
        #         self.m_stability = False
        #         self.m_Kp = 0.18
        #         self.m_Ki = 0.008
        #         self.m_Kd = 0.08
        #     else:
        #         self.m_cur_increase = -1
        #         self.m_stability = True
        #         self.m_steering_variant = []
        # else:
        #     if self.m_stability == True:
        #         if self.m_Kp < 0.18:
        #             self.m_Kp += self.m_kp_decrease
        #         if self.m_Ki < 0.008:
        #             self.m_Ki += self.m_ki_decrease
        #         if self.m_Kd < 0.08:
        #             self.m_Kd += self.m_kd_decrease
        #         self.m_Kp = 0.18
        #         self.m_Ki = 0.008
        #         self.m_Kd = 0.08

        # self.m_Kp = 0.3
        # self.m_Ki = 0.003
        # self.m_Kd = 0.03

        self.m_output      = self.m_p_value * self.m_Kp + self.m_i_value * self.m_Ki + self.m_d_value * self.m_Kd
        self.m_delta_time  = delta_time
        self.m_delta_error = delta_error
        self.m_last_time   = cur_time
        self.m_last_error  = error

    def update_offset(self, offset_value, angle_value, error, delta_time, delta_error, cur_time):

        self.m_p_value = error
        self.m_i_value = min(max(error * delta_time, -self.m_max_integral), self.m_max_integral)
        self.m_d_value = delta_error / delta_time if delta_time > 0 else 0.0

        # if abs(offset_value) <= 0.1 and abs(angle_value) <= 0.1745:
        #     self.m_Kp = 0.3
        # else:
        #     self.m_Kp += self.m_kp_decrease

        self.m_output      = self.m_p_value * self.m_Kp + self.m_i_value * self.m_Ki + self.m_d_value * self.m_Kd
        self.m_delta_time  = delta_time
        self.m_delta_error = delta_error
        self.m_last_time   = cur_time
        self.m_last_error  = error

    def update_speed(self, angle_value, speed_value, error, delta_time, delta_error, cur_time):

        self.m_steering_variant.append(angle_value)
        self.m_p_value = error
        self.m_i_value = min(max(error * delta_time, -self.m_max_integral), self.m_max_integral)
        self.m_d_value = delta_error / delta_time if delta_time > 0 else 0.0

        self.m_output      = self.m_p_value * self.m_Kp + self.m_i_value * self.m_Ki + self.m_d_value * self.m_Kd
        self.m_delta_time  = delta_time
        self.m_delta_error = delta_error
        self.m_last_time   = cur_time
        self.m_last_error  = error

        if len(self.m_steering_variant) >= 5:
            probability = self.stability_of_wheel(
                self.m_steering_variant[(len(self.m_steering_variant) - 5):len(self.m_steering_variant)])
            probability2 = self.stability_of_wheel_2(
                self.m_steering_variant[(len(self.m_steering_variant) - 5):len(self.m_steering_variant)])

            if probability >= 0.2 or probability2>0.7:
                increase_value = 0
                if self.m_cur_increase != -1:
                    increase_value = math.fabs(self.m_cur_increase - 0.15)
                    self.m_cur_increase = -1
                #print "break"

                self.m_output = self.m_output - self.m_brakes - increase_value
                self.m_stability = False
            else:
                self.m_cur_increase = -1
                self.m_stability = True
                self.m_steering_variant = []
        else:
            if self.m_stability == True:
                #print "speed up"
                if self.m_cur_increase == -1:
                    self.m_cur_increase = self.m_velocity
                else:
                    self.m_cur_increase += self.m_increase
                self.m_output += self.m_cur_increase

    def update(self, update_value, second_value, cur_time = None):

        if cur_time is None:
            cur_time = time()

        error       = self.m_set_point - update_value
        delta_time  = cur_time - self.m_last_time
        delta_error = error - self.m_last_error

        if delta_time >= self.m_min_interval:

            if self.m_mode == 'wheel':
                self.update_wheel(update_value, error, delta_time, delta_error, cur_time)

            elif self.m_mode == 'offset':
                self.update_offset(update_value, second_value, error, delta_time, delta_error, cur_time)

            elif self.m_mode == 'speed':
                self.update_speed(update_value, second_value, error, delta_time, delta_error, cur_time)

        return self.m_output

    def reset(self, last_time = None, set_point = 0.0):

        self.m_Kp = self.m_ori_Kp
        self.m_Ki = self.m_ori_Ki
        self.m_Kd = self.m_ori_Kd

        self.m_set_point   = set_point
        self.m_last_time   = last_time if last_time is not None else time()
        self.m_p_value     = 0.0
        self.m_i_value     = 0.0
        self.m_d_value     = 0.0
        self.m_delta_time  = 0.0
        self.m_delta_error = 0.0
        self.m_last_error  = 0.0
        self.m_output      = 0.0

        self.m_steering_variant = []
        self.m_stability    = True
        self.m_cur_increase = -1

    def assign_set_point(self, set_point):

        self.m_set_point = set_point

    def assing_pid_Kp(self, pid_Kp):

        self.m_Kp = pid_Kp

    def assing_pid_Ki(self, pid_Ki):

        self.m_Ki = pid_Ki

    def assing_pid_Kd(self, pid_Kd):

        self.m_Kd = pid_Kd

    def get_set_point(self):

        return self.m_set_point

    def get_pid_Kp(self):

        return self.m_p_value

    def get_pid_Ki(self):

        return self.m_i_value

    def get_pid_Kd(self):

        return self.m_d_value

    def get_delta_time(self):

        return self.m_delta_time

    def get_delta_error(self):

        return self.m_delta_error

    def get_last_error(self):

        return self.m_last_error

    def get_last_time(self):

        return self.m_last_time

    def get_output(self):

        return self.m_output


class twQTeamImageProcessor(object):

    def __init__(self, track_mode=0):

        self.m_img_width     = 160
        self.m_img_height    = 120
        self.m_center_img_x  = 80
        self.m_crop_ratio    = 0.55
        self.m_crop_ratios   = (self.m_crop_ratio , 1.0)
        self.m_crop_slice    = slice(*(int(cnt * self.m_img_height) for cnt in self.m_crop_ratios))
        self.m_crop_value    = int(self.m_img_height * self.m_crop_ratio)
        self.m_max_gap       = self.m_img_width + self.m_img_height
        self.m_offset_ratio  = 0.3
        self.m_offset_y      = self.m_offset_ratio * self.m_crop_value
        self.m_last_steering = 0
        self.m_track_mode    = track_mode
        self.m_initial_track = track_mode
        self.m_debug_mode    = FLAG_SHOW_WINDOW
        self.m_pre_img       = None
        self.m_draw_img      = None
        self.m_nSplit        = 40
        self.m_scan_lines    = []
        for cnt in range(0, self.m_nSplit):
            loc_y = self.m_img_height * float(cnt) / self.m_nSplit
            if loc_y < int(self.m_img_height * self.m_crop_ratio) or loc_y > self.m_img_height:
                continue
            else:
                loc_y -= int(self.m_img_height * self.m_crop_ratio)
                self.m_scan_lines.append(int(loc_y))

    @staticmethod
    def radiusTodegree(radius):
        return radius / np.pi * 180.0

    @staticmethod
    def degreeToradius(degree):
        return degree / 180.0 * np.pi

    @staticmethod
    def rgbTobgr(img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def showImage(self, img, name="image", scale=1.0):

        if scale and scale != 1.0:
            newsize = (int(img.shape[1]*scale), int(img.shape[0]*scale))
            img = cv2.resize(img, newsize, interpolation=cv2.INTER_LINEAR)

        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(name, img)
        cv2.moveWindow(name, 170, 170)
        cv2.waitKey(1)

    def setTrackMode(self, track_mdoe):

        # mode 0 represents |bW|r|b|r|g|b|g|bW|
        # mode 1 represents |bW|g|b|g|r|b|r|bW|
        # mode 2 represents |bW|r|b|r|yW|
        # mode 3 represents |yW|g|b|g|bW|
        # mode 4 represents |bW|g|b|g|yW|
        # mode 5 represents |yW|r|b|r|bW|
        self.m_track_mode = track_mdoe

    def getTrackMode(self):

        return self.m_track_mode

    def cropImage(self, img):

        crop_img = img[self.m_crop_slice, :, :]

        return crop_img

    def brightenImage(self, img):

        max_pixel = img.max()
        max_pixel = max_pixel if max_pixel != 0 else 255

        brighten_img = img * (255 / max_pixel)
        brighten_img = np.clip(brighten_img, 0, 255)
        brighten_img = np.array(brighten_img, dtype=np.uint8)

        return brighten_img

    def flattenImage(self, img):

        bChn_img, gChn_img, rChn_img = cv2.split(img)
        max_pixel = np.maximum(np.maximum(bChn_img, gChn_img), rChn_img)
        b_filter  = (bChn_img == max_pixel) & (bChn_img >= 120) & (gChn_img < 150) & (rChn_img < 150)
        g_filter  = (gChn_img == max_pixel) & (gChn_img >= 120) & (bChn_img < 150) & (rChn_img < 150)
        r_filter  = (rChn_img == max_pixel) & (rChn_img >= 120) & (bChn_img < 150) & (gChn_img < 150)
        y_filter  = ((bChn_img >= 128) & (gChn_img >= 128) & (rChn_img < 100))
        bChn_img[y_filter]            = 255
        gChn_img[y_filter]            = 255
        rChn_img[np.invert(y_filter)] = 0
        rChn_img[r_filter], rChn_img[np.invert(r_filter)] = 255, 0
        bChn_img[b_filter], bChn_img[np.invert(b_filter)] = 255, 0
        gChn_img[g_filter], gChn_img[np.invert(g_filter)] = 255, 0
        # flat_img = cv2.merge([bChn_img, gChn_img, rChn_img])

        return bChn_img, gChn_img, rChn_img #flat_img

    def getExtWallImage(self, img):

        lower_black = np.array([0, 0, 0])
        upper_black = np.array([50, 50, 50])
        bWall_img   = cv2.inRange(img, lower_black, upper_black)

        return bWall_img

    def getForkWallImage(self, bWall_img, bChn_img, gChn_img, rChn_img):

        # bChn_img, gChn_img, rChn_img = cv2.split(flat_img)
        yWall_img = np.invert(bChn_img | gChn_img | rChn_img | bWall_img)

        return yWall_img

    def getWallImage(self, bWall_img, yWall_img):

        allWall_img = (bWall_img | yWall_img)

        return allWall_img

    def mergeBlobs(self, blob_stats, blob_count):

        blob_stats.sort(key=lambda var: -var[4])
        blob_stats = np.asarray(blob_stats)
        idx_remove = []

        for cnt1 in range(blob_count - 1):
            for cnt2 in range(cnt1 + 1, blob_count):

                for idx in idx_remove:
                    if idx == cnt2:
                        continue

                blob_1 = blob_stats[cnt1]
                blob_2 = blob_stats[cnt2]
                dist_thrd = 10
                bOverlapBlob = ((abs(blob_1[0] - blob_2[0]) <= dist_thrd and abs(blob_1[0] - blob_2[2]) <= dist_thrd) or
                                (abs(blob_1[2] - blob_2[0]) <= dist_thrd and abs(blob_1[2] - blob_2[2]) <= dist_thrd)) and \
                               ((abs(blob_1[1] - blob_2[1]) <= dist_thrd and abs(blob_1[1] - blob_2[3]) <= dist_thrd) or
                                (abs(blob_1[3] - blob_2[1]) <= dist_thrd and abs(blob_1[3] - blob_2[3]) <= dist_thrd))
                bWithinBlob  = blob_1[0] <= blob_2[0] and blob_1[2] >= blob_2[2] and \
                               blob_1[1] <= blob_2[1] and blob_1[3] >= blob_2[3]
                bIsNearBlob  = (abs(blob_1[0] - blob_2[0]) <= dist_thrd or abs(blob_1[0] - blob_2[2]) <= dist_thrd or
                                abs(blob_1[2] - blob_2[0]) <= dist_thrd or abs(blob_1[2] - blob_2[2]) <= dist_thrd) and \
                               (abs(blob_1[1] - blob_2[1]) <= dist_thrd or abs(blob_1[1] - blob_2[3]) <= dist_thrd or
                                abs(blob_1[3] - blob_2[1]) <= dist_thrd or abs(blob_1[3] - blob_2[3]) <= dist_thrd)

                if bOverlapBlob or bWithinBlob or bIsNearBlob:
                    loc_left_x  = blob_1[0] if blob_1[0] < blob_2[0] else blob_2[0]
                    loc_left_y  = blob_1[1] if blob_1[1] < blob_2[1] else blob_2[1]
                    loc_right_x = blob_1[2] if blob_1[2] > blob_2[2] else blob_2[2]
                    loc_right_y = blob_1[3] if blob_1[3] > blob_2[3] else blob_2[3]
                    blob_stats[cnt1][0] = loc_left_x
                    blob_stats[cnt1][1] = loc_left_y
                    blob_stats[cnt1][2] = loc_right_x
                    blob_stats[cnt1][3] = loc_right_y
                    blob_stats[cnt1][4] += blob_2[4]
                    idx_remove.append(cnt2)

        if len(idx_remove) != 0:
            new_blob_stats = []
            for cnt1 in range(0, len(blob_stats)):
                bBeRemoved = False
                for cnt2 in idx_remove:
                    if cnt1 == cnt2:
                        bBeRemoved = True
                        break
                if bBeRemoved == False:
                    new_blob_stats.append(blob_stats[cnt1])

            return new_blob_stats
        else:
            return blob_stats

    def trafficSignDetection(self, img):

        blob_img  = img.copy()
        crop_img  = blob_img[0:80, :, :]
        hsv_img   = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        lower_red = np.array([-10, 100, 10])
        upper_red = np.array([10, 255, 255])
        mask_red  = cv2.inRange(hsv_img, lower_red, upper_red)
        red_img   = cv2.bitwise_and(crop_img, crop_img, mask=mask_red)

        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_red, connectivity=8, ltype=cv2.CV_32S)

        min_thd    = 2
        max_thd    = 1000
        max_width  = 60
        min_width  = 5
        blob_stats = []
        for cnt in range(nlabels):
            bIsTrafficSign = stats[cnt][4] >= min_thd and stats[cnt][4] <= max_thd and \
                             stats[cnt][2] <= max_width and stats[cnt][2] >= min_width
            if bIsTrafficSign == True:
                loc_left = (stats[cnt][0], stats[cnt][1])
                loc_right = (stats[cnt][0] + stats[cnt][2], stats[cnt][1] + stats[cnt][3])
                blob_stats.append((loc_left[0], loc_left[1], loc_right[0], loc_right[1], stats[cnt][4]))

        if len(blob_stats) > 1:
            blob_stats = self.mergeBlobs(blob_stats, len(blob_stats))

        return blob_stats #blob_img

    def processImage(self, img):

        crop_img    = self.cropImage(img)
        # correct_img = self.brightenImage(crop_img)
        correct_img = crop_img
        # flat_img    = self.flattenImage(correct_img)
        bChn_img, gChn_img, rChn_img = self.flattenImage(correct_img)
        bWall_img   = self.getExtWallImage(correct_img)
        yWall_img   = self.getForkWallImage(bWall_img, bChn_img, gChn_img, rChn_img)
        allWall_img = self.getWallImage(bWall_img, yWall_img)

        proc_img = []
        proc_img.append(bChn_img)
        proc_img.append(gChn_img)
        proc_img.append(rChn_img)
        proc_img.append(bWall_img)
        proc_img.append(yWall_img)
        proc_img.append(allWall_img)

        return proc_img

    def getEdgePoint(self, line):

        pos_on    = []
        pos_off   = []
        pos_chg = np.where(line[:-1] != line[1:])[0]
        pos_chg = np.asarray(pos_chg)

        thrd_linked = 10
        idx_remove = []
        for cnt in range(0, int(len(pos_chg)*0.5), 2):
            if abs(pos_chg[cnt] - pos_chg[cnt+1]) < thrd_linked:
                idx_remove.append(cnt)
                idx_remove.append(cnt+1)
        pos_chg = np.delete(pos_chg, idx_remove)

        thrd_boundary = 3
        idx_remove = []
        for cnt in pos_chg:
            if cnt < thrd_boundary or cnt >= self.m_img_width-thrd_boundary:
                idx_remove.append(cnt)
        pos_chg = np.delete(pos_chg, idx_remove)

        for cnt in pos_chg:
            if line[cnt] == True:
                pos_off.append(cnt)
            else:
                pos_on.append(cnt)

        return pos_on, pos_off

    def getEdgeLine(self, loc_y, posEdgePointLeft, posEdgePointRight, line_edges, line_noise):

        idx_remove_left = []
        idx_remove_right = []
        min_gap = 20
        max_gap = self.m_max_gap
        for cnt1 in range(0, len(posEdgePointLeft)):
            dist_len = max_gap
            dist_idx = -1
            for cnt2 in range(0, len(posEdgePointRight)):
                if abs(posEdgePointLeft[cnt1] - posEdgePointRight[cnt2]) < dist_len:
                    dist_len = abs(posEdgePointLeft[cnt1] - posEdgePointRight[cnt2])
                    dist_idx = cnt2

            if dist_len < min_gap:
                idx_remove_left.append(cnt1)
                idx_remove_right.append(dist_idx)
                loc_x = int((posEdgePointLeft[cnt1] + posEdgePointRight[dist_idx]) * 0.5) #/ 2
                line_edges.append((loc_x, loc_y))
                continue
            else:
                line_noise.append((posEdgePointLeft[cnt1], loc_y))

        new_posEdgePointLeft = np.delete(posEdgePointLeft, idx_remove_left)
        new_posEdgePointRight = np.delete(posEdgePointRight, idx_remove_right)

        return new_posEdgePointLeft, new_posEdgePointRight

    def findBestEgdeLine(self, line_edges, start_index):

        max_len = 0
        max_idx = 0
        for cnt in range(1, len(line_edges)-1):
            len_edge = len(line_edges[cnt])
            if len_edge > max_len:
                max_len = len_edge
                max_idx = cnt
        line_best     = line_edges[max_idx]
        line_best_idx = max_idx
        if max_len < 5:
            line_best     = []
            line_best_idx = -1

        return line_best, line_best_idx

    def doEdgeDetection(self, img, proc_img):

        bTrack_img = proc_img[0] > 0
        gTrack_img = proc_img[1] > 0
        rTrack_img = proc_img[2] > 0
        b_Wall_img = proc_img[3] > 0
        y_Wall_img = proc_img[4] > 0
        bgWall_img = proc_img[5] > 0
        line_edges = None
        if self.m_track_mode == 0 or self.m_track_mode == 1:
            line_edges = [[],[],[],[],[],[],[]]
        else:
            line_edges = [[],[],[],[]]
        line_noise = []
        line_rev   = [[],[],[]]
        for cnt in range(0, len(self.m_scan_lines)):
            loc_y = self.m_scan_lines[cnt]
            bTrack_posOn, bTrack_posOff = self.getEdgePoint(bTrack_img[loc_y, :])
            gTrack_posOn, gTrack_posOff = self.getEdgePoint(gTrack_img[loc_y, :])
            rTrack_posOn, rTrack_posOff = self.getEdgePoint(rTrack_img[loc_y, :])
            b_Wall_posOn, b_Wall_posOff = self.getEdgePoint(b_Wall_img[loc_y, :])
            y_Wall_posOn, y_Wall_posOff = self.getEdgePoint(y_Wall_img[loc_y, :])
            bgWall_posOn, bgWall_posOff = self.getEdgePoint(bgWall_img[loc_y, :])

            # mode 0 represents |bW|r|b|r|g|b|g|bW|
            if self.m_track_mode == 0:
                bgWall_posOff, gTrack_posOn = self.getEdgeLine(loc_y, bgWall_posOff, gTrack_posOn, line_rev[0], line_noise)
                gTrack_posOff, rTrack_posOn = self.getEdgeLine(loc_y, gTrack_posOff, rTrack_posOn, line_rev[1], line_noise)
                rTrack_posOff, bgWall_posOn = self.getEdgeLine(loc_y, rTrack_posOff, bgWall_posOn, line_rev[2], line_noise)
                bgWall_posOff, rTrack_posOn = self.getEdgeLine(loc_y, bgWall_posOff, rTrack_posOn, line_edges[0], line_noise)
                rTrack_posOff, bTrack_posOn = self.getEdgeLine(loc_y, rTrack_posOff, bTrack_posOn, line_edges[1], line_noise)
                bTrack_posOff, rTrack_posOn = self.getEdgeLine(loc_y, bTrack_posOff, rTrack_posOn, line_edges[2], line_noise)
                rTrack_posOff, gTrack_posOn = self.getEdgeLine(loc_y, rTrack_posOff, gTrack_posOn, line_edges[3], line_noise)
                gTrack_posOff, bTrack_posOn = self.getEdgeLine(loc_y, gTrack_posOff, bTrack_posOn, line_edges[4], line_noise)
                bTrack_posOff, gTrack_posOn = self.getEdgeLine(loc_y, bTrack_posOff, gTrack_posOn, line_edges[5], line_noise)
                gTrack_posOff, bgWall_posOn = self.getEdgeLine(loc_y, gTrack_posOff, bgWall_posOn, line_edges[6], line_noise)

            # mode 1 represents |bW|g|b|g|r|b|r|bW|
            elif self.m_track_mode == 1:
                bgWall_posOff, rTrack_posOn = self.getEdgeLine(loc_y, bgWall_posOff, rTrack_posOn, line_rev[0], line_noise)
                rTrack_posOff, gTrack_posOn = self.getEdgeLine(loc_y, rTrack_posOff, gTrack_posOn, line_rev[1], line_noise)
                gTrack_posOff, bgWall_posOn = self.getEdgeLine(loc_y, gTrack_posOff, bgWall_posOn, line_rev[2], line_noise)
                bgWall_posOff, gTrack_posOn = self.getEdgeLine(loc_y, bgWall_posOff, gTrack_posOn, line_edges[0], line_noise)
                gTrack_posOff, bTrack_posOn = self.getEdgeLine(loc_y, gTrack_posOff, bTrack_posOn, line_edges[1], line_noise)
                bTrack_posOff, gTrack_posOn = self.getEdgeLine(loc_y, bTrack_posOff, gTrack_posOn, line_edges[2], line_noise)
                gTrack_posOff, rTrack_posOn = self.getEdgeLine(loc_y, gTrack_posOff, rTrack_posOn, line_edges[3], line_noise)
                rTrack_posOff, bTrack_posOn = self.getEdgeLine(loc_y, rTrack_posOff, bTrack_posOn, line_edges[4], line_noise)
                bTrack_posOff, rTrack_posOn = self.getEdgeLine(loc_y, bTrack_posOff, rTrack_posOn, line_edges[5], line_noise)
                rTrack_posOff, bgWall_posOn = self.getEdgeLine(loc_y, rTrack_posOff, bgWall_posOn, line_edges[6], line_noise)

            # mode 2 represents |bW|r|b|r|yW|
            elif self.m_track_mode == 2:
                y_Wall_posOff, rTrack_posOn = self.getEdgeLine(loc_y, y_Wall_posOff, rTrack_posOn, line_rev[0], line_noise)
                rTrack_posOff, b_Wall_posOn = self.getEdgeLine(loc_y, rTrack_posOff, b_Wall_posOn, line_rev[1], line_noise)
                b_Wall_posOff, rTrack_posOn = self.getEdgeLine(loc_y, b_Wall_posOff, rTrack_posOn, line_edges[0], line_noise)
                rTrack_posOff, bTrack_posOn = self.getEdgeLine(loc_y, rTrack_posOff, bTrack_posOn, line_edges[1], line_noise)
                bTrack_posOff, rTrack_posOn = self.getEdgeLine(loc_y, bTrack_posOff, rTrack_posOn, line_edges[2], line_noise)
                rTrack_posOff, y_Wall_posOn = self.getEdgeLine(loc_y, rTrack_posOff, y_Wall_posOn, line_edges[3], line_noise)

            # mode 3 represents |yW|g|b|g|bW|
            elif self.m_track_mode == 3:
                b_Wall_posOff, gTrack_posOn = self.getEdgeLine(loc_y, b_Wall_posOff, gTrack_posOn, line_rev[0], line_noise)
                gTrack_posOff, y_Wall_posOn = self.getEdgeLine(loc_y, gTrack_posOff, y_Wall_posOn, line_rev[1], line_noise)
                y_Wall_posOff, gTrack_posOn = self.getEdgeLine(loc_y, y_Wall_posOff, gTrack_posOn, line_edges[0], line_noise)
                gTrack_posOff, bTrack_posOn = self.getEdgeLine(loc_y, gTrack_posOff, bTrack_posOn, line_edges[1], line_noise)
                bTrack_posOff, gTrack_posOn = self.getEdgeLine(loc_y, bTrack_posOff, gTrack_posOn, line_edges[2], line_noise)
                gTrack_posOff, b_Wall_posOn = self.getEdgeLine(loc_y, gTrack_posOff, b_Wall_posOn, line_edges[3], line_noise)

            # mode 4 represents |bW|g|b|g|yW|
            elif self.m_track_mode == 4:
                y_Wall_posOff, gTrack_posOn = self.getEdgeLine(loc_y, y_Wall_posOff, gTrack_posOn, line_rev[0], line_noise)
                gTrack_posOff, b_Wall_posOn = self.getEdgeLine(loc_y, gTrack_posOff, b_Wall_posOn, line_rev[1], line_noise)
                b_Wall_posOff, gTrack_posOn = self.getEdgeLine(loc_y, b_Wall_posOff, gTrack_posOn, line_edges[0], line_noise)
                gTrack_posOff, bTrack_posOn = self.getEdgeLine(loc_y, gTrack_posOff, bTrack_posOn, line_edges[1], line_noise)
                bTrack_posOff, gTrack_posOn = self.getEdgeLine(loc_y, bTrack_posOff, gTrack_posOn, line_edges[2], line_noise)
                gTrack_posOff, y_Wall_posOn = self.getEdgeLine(loc_y, gTrack_posOff, y_Wall_posOn, line_edges[3], line_noise)

            # mode 5 represents |yW|r|b|r|bW|
            elif self.m_track_mode == 5:
                b_Wall_posOff, rTrack_posOn = self.getEdgeLine(loc_y, b_Wall_posOff, rTrack_posOn, line_rev[0], line_noise)
                rTrack_posOff, y_Wall_posOn = self.getEdgeLine(loc_y, rTrack_posOff, y_Wall_posOn, line_rev[1], line_noise)
                y_Wall_posOff, rTrack_posOn = self.getEdgeLine(loc_y, y_Wall_posOff, rTrack_posOn, line_edges[0], line_noise)
                rTrack_posOff, bTrack_posOn = self.getEdgeLine(loc_y, rTrack_posOff, bTrack_posOn, line_edges[1], line_noise)
                bTrack_posOff, rTrack_posOn = self.getEdgeLine(loc_y, bTrack_posOff, rTrack_posOn, line_edges[2], line_noise)
                rTrack_posOff, b_Wall_posOn = self.getEdgeLine(loc_y, rTrack_posOff, b_Wall_posOn, line_edges[3], line_noise)

        line_results = []
        line_results.append(line_edges)
        line_results.append(line_noise)
        line_results.append(line_rev)

        return line_results

    def doEdgeDetectionCenterLine(self, proc_img):

        gTrack_img = proc_img[1] > 0
        rTrack_img = proc_img[2] > 0
        line_edges = []
        line_noise = []
        if self.m_initial_track == 0:
            for cnt in range(0, len(self.m_scan_lines)):
                loc_y = self.m_scan_lines[cnt]
                gTrack_posOn, gTrack_posOff = self.getEdgePoint(gTrack_img[loc_y, :])
                rTrack_posOn, rTrack_posOff = self.getEdgePoint(rTrack_img[loc_y, :])
                rTrack_posOff, gTrack_posOn = self.getEdgeLine(loc_y, rTrack_posOff, gTrack_posOn, line_edges, line_noise)
        else:
            for cnt in range(0, len(self.m_scan_lines)):
                loc_y = self.m_scan_lines[cnt]
                gTrack_posOn, gTrack_posOff = self.getEdgePoint(gTrack_img[loc_y, :])
                rTrack_posOn, rTrack_posOff = self.getEdgePoint(rTrack_img[loc_y, :])
                gTrack_posOff, rTrack_posOn = self.getEdgeLine(loc_y, gTrack_posOff, rTrack_posOn, line_edges, line_noise)

        return line_edges

    def findSteeringAngle(self, img, proc_img, start_index):

        line_results = self.doEdgeDetection(img, proc_img)
        line_best, line_best_idx = self.findBestEgdeLine(line_results[0], start_index)
        line_results.append(line_best)
        line_results.append(line_best_idx)

        steering_radian = 0.0

        return steering_radian, line_results

    def convertToUnityAngle(self, radian):

        radian = -1 * radian
        angle = float(radian) * 57.29578
        angle = 90 - angle

        return angle

    def convertToUnityRadian(self, angle):

        radian = float(angle) * 0.01745

        return radian

    def findOffsetDistance(self, line_best):

        if line_best == []:
            return 0

        offset_y = self.m_offset_y
        min_dist = abs(line_best[0][1] - offset_y)
        min_idx  = 0
        for cnt in range(1, len(line_best)):
            if abs(line_best[cnt][1] - offset_y) < min_dist:
                min_dist = abs(line_best[cnt][1] - offset_y)
                min_idx  = cnt

        offset = line_best[min_idx][0] - self.m_center_img_x
        offset = float(offset) * 0.0125

        return offset


class twQTeamDrive(object):

    def __init__(self, car, record_folder = None):
        self.m_steering_Kp           = 0.2
        self.m_steering_Ki           = 0.008
        self.m_steering_Kd           = 0.08
        self.m_steering_max_integral = 40
        self.m_distance_Kp           = 0.3
        self.m_distance_Ki           = 0.003
        self.m_distance_Kd           = 0.03
        self.m_distance_max_integral = 100
        self.m_throttle_Kp           = 0.02
        self.m_throttle_Ki           = 0.005
        self.m_throttle_Kd           = 0.02
        self.m_throttle_max_integral = 0.5
        self.m_max_steering_history  = 3
        self.m_max_distance_history  = 3
        self.m_max_throttle_history  = 3
        self.m_default_speed         = 0.5
        self.m_debug_mode            = FLAG_SHOW_WINDOW

        self.m_record_folder         = record_folder
        self.m_steering_pid          = twQTeamPID(mode='wheel', Kp=self.m_steering_Kp, Ki=self.m_steering_Ki, Kd=self.m_steering_Kd, max_integral=self.m_steering_max_integral)
        self.m_distance_pid          = twQTeamPID(mode='offset', Kp=self.m_distance_Kp, Ki=self.m_distance_Ki, Kd=self.m_distance_Kd, max_integral=self.m_distance_max_integral)
        self.m_throttle_pid          = twQTeamPID(mode='speed', Kp=self.m_throttle_Kp, Ki=self.m_throttle_Ki, Kd=self.m_throttle_Kd, max_integral=self.m_throttle_max_integral)
        self.m_throttle_pid.assign_set_point(self.m_default_speed)
        self.m_steering_history      = []
        self.m_distance_history      = []
        self.m_throttle_history      = []
        self.m_car                   = car
        self.m_car.register(self)
        self.m_count_save_img        = 0
        self.m_twQTeamImageProcessor = twQTeamImageProcessor(track_mode=0)
        self.m_current_lap           = 1
        self.m_last_elapsed          = 0
        self.m_last_lap_elapsed      = 999
        self.m_best_lap_elapsed      = 999
        self.m_best_lap_elapsed_idx  = 0
        self.m_last_track_index      = 0
        self.m_last_angle            = 0
        self.m_const_radian_10       = 0.01745 * 10
        self.m_const_radian_15       = 0.01745 * 15
        self.m_const_radian_20       = 0.01745 * 20
        self.m_const_radian_25       = 0.01745 * 25
        self.m_is_in_fork_track      = False
        self.m_cnt_delay_frame       = 0
        self.m_seen_fork_left_sign   = False
        self.m_seen_fork_right_sign  = False
        self.m_cnt_fork_left_sign    = 0
        self.m_cnt_fork_right_sign   = 0
        self.m_last_fork_time        = 0

        self.m_cnt_turn_left_sign    = 0
        self.m_cnt_turn_right_sign   = 0
        self.m_last_turn_time        = 0
        self.m_cnt_u_turn_left_sign  = 0
        self.m_cnt_u_turn_right_sign = 0
        self.m_last_u_turn_time      = 0

        self.m_time_traffic_sign     = 0
        self.m_cnt_order             = 0
        self.m_signInfo              = []
        self.m_signInfo.append('')

        self.m_cnt_speed_zero        = 0
        self.m_is_find_line          = False

    def getSteering(self, src_img, proc_img, angle_index):

        if self.m_twQTeamImageProcessor.getTrackMode() == 2:
            start_index = 2
        else:
            start_index = 1
        cur_radian, line_results = self.m_twQTeamImageProcessor.findSteeringAngle(src_img, proc_img, start_index)
        line_edges    = line_results[0]
        line_noise    = line_results[1]
        line_rev      = line_results[2]
        line_best     = line_results[3]
        line_best_idx = line_results[4]
        cur_angle     = self.m_twQTeamImageProcessor.convertToUnityAngle(cur_radian)
        cur_radian    = self.m_twQTeamImageProcessor.convertToUnityRadian(cur_angle)

        slop_angle = 0.0
        slop_radian = 0.0
        if line_best != []:
            delta_X     = float(line_best[angle_index][0] - self.m_twQTeamImageProcessor.m_center_img_x)
            delta_Y     = float(line_best[angle_index][1] + self.m_twQTeamImageProcessor.m_crop_value - self.m_twQTeamImageProcessor.m_img_height)
            slop_radian = math.atan2(delta_Y, delta_X)
            slop_angle  = self.m_twQTeamImageProcessor.convertToUnityAngle(slop_radian)
            slop_radian = self.m_twQTeamImageProcessor.convertToUnityRadian(slop_angle)
        steering_angle = self.m_steering_pid.update(-slop_radian, None)

        if abs(slop_angle) > 15:
            self.m_distance_pid.assing_pid_Kp(0.05)
            self.m_distance_pid.assing_pid_Ki(0.0005)
            self.m_distance_pid.assing_pid_Kd(0.005)
        else:
            self.m_distance_pid.assing_pid_Kp(self.m_distance_pid.m_ori_Kp)
            self.m_distance_pid.assing_pid_Ki(self.m_distance_pid.m_ori_Ki)
            self.m_distance_pid.assing_pid_Kd(self.m_distance_pid.m_ori_Kd)

        offset_distance = self.m_twQTeamImageProcessor.findOffsetDistance(line_best)
        offset_steering = self.m_distance_pid.update(-offset_distance, slop_radian)

        return line_edges, line_rev, line_best_idx, slop_angle, slop_radian, steering_angle, offset_distance, offset_steering

    def adjustSteeringInFullTrack(self, line_edges, line_best_idx, slop_angle, steering_angle, offset_distance, offset_steering):

        if line_best_idx != 1 and line_best_idx != 5:
            if line_best_idx == 3:
                if abs(offset_distance) > 0.5:
                    offset_steering = 0
                elif abs(offset_distance) > 0.4:
                    offset_steering *= 0.1
                elif abs(offset_distance) > 0.3:
                    offset_steering *= 0.3
                elif abs(offset_distance) > 0.2:
                    offset_steering *= 0.5
                elif abs(offset_distance) > 0.1:
                    offset_steering *= 0.7
            if line_best_idx == 2 or line_best_idx == 4:
                if abs(offset_distance) > 0.4:
                    offset_steering = 0
                elif abs(offset_distance) > 0.3:
                    offset_steering *= 0.1
                elif abs(offset_distance) > 0.2:
                    offset_steering *= 0.3
                elif abs(offset_distance) > 0.1:
                    offset_steering *= 0.5

        if abs(slop_angle) < 15:
            output_steering = offset_steering
        else:
            output_steering = steering_angle + offset_steering

        if line_best_idx == 1:
            output_steering += self.m_const_radian_10
        if line_best_idx == 5:
            output_steering -= self.m_const_radian_10

        if len(line_edges[0]) >= 3:
            output_steering += self.m_const_radian_15
        if len(line_edges[6]) >= 3:
            output_steering -= self.m_const_radian_15

        return output_steering

    def adjustSteeringInForkTrack(self, line_edges, line_best_idx, slop_angle, steering_angle, offset_distance, offset_steering):

        if abs(slop_angle) < 15:
            output_steering = offset_steering
        else:
            output_steering = steering_angle + offset_steering

        return output_steering

    def getThrottleInFullTrack(self):

        output_throttle = 0.2

        return output_throttle

    def getThrottleInForkTrack(self):

        output_throttle = 0.02

        return output_throttle

    def adjustControlBySharpTurnInFullTrack(self, line_best_idx, slop_angle, output_steering, output_throttle):

        if line_best_idx == -1:

            if self.m_last_track_index == 1 or self.m_last_track_index == 5:
                if abs(self.m_last_angle) > 40:
                    output_steering = self.m_twQTeamImageProcessor.convertToUnityRadian(self.m_last_angle * 4)
                    output_throttle = -0.7
                else:
                    output_steering = self.m_twQTeamImageProcessor.convertToUnityRadian(self.m_last_angle * 3)
                    output_throttle = -0.6

            if self.m_last_track_index == 2 or self.m_last_track_index == 4:
                if abs(self.m_last_angle) > 40:
                    output_steering = self.m_twQTeamImageProcessor.convertToUnityRadian(self.m_last_angle * 4)
                    output_throttle = -0.7

                elif abs(self.m_last_angle) > 30:
                    output_steering = self.m_twQTeamImageProcessor.convertToUnityRadian(self.m_last_angle * 3)
                    output_throttle = -0.6
                else:
                    output_steering = self.m_twQTeamImageProcessor.convertToUnityRadian(self.m_last_angle * 1)
                    output_throttle = -0.5

            else:
                output_steering = self.m_twQTeamImageProcessor.convertToUnityRadian(self.m_last_angle)
                output_throttle = -0.4

        else:
            if self.m_last_track_index != line_best_idx:
                if (self.m_last_track_index == line_best_idx + 1) or (self.m_last_track_index + 1 == line_best_idx):
                    output_throttle = output_throttle
                else:
                    output_throttle = -0.01

        self.m_last_track_index = line_best_idx
        self.m_last_angle       = slop_angle

        return output_steering, output_throttle

    def adjustControlBySharpTurnInForkTrack(self, line_best_idx, slop_angle, output_steering, output_throttle):

        if line_best_idx == -1:

            if self.m_last_track_index == 1 or self.m_last_track_index == 2:
                if abs(self.m_last_angle) > 40:
                    output_steering = self.m_twQTeamImageProcessor.convertToUnityRadian(self.m_last_angle * 4)
                    output_throttle = -0.3

                elif abs(self.m_last_angle) > 30:
                    output_steering = self.m_twQTeamImageProcessor.convertToUnityRadian(self.m_last_angle * 3)
                    output_throttle = -0.3
                else:
                    output_steering = self.m_twQTeamImageProcessor.convertToUnityRadian(self.m_last_angle * 1)
                    output_throttle = -0.2

            else:
                output_steering = self.m_twQTeamImageProcessor.convertToUnityRadian(self.m_last_angle)
                output_throttle = -0.1

        else:
            if self.m_last_track_index != line_best_idx:
                if (self.m_last_track_index == line_best_idx + 1) or (self.m_last_track_index + 1 == line_best_idx):
                    output_throttle = output_throttle
                else:
                    output_throttle = -0.001

        self.m_last_track_index = line_best_idx
        self.m_last_angle       = slop_angle

        return output_steering, output_throttle

    def on_dashboard(self, src_img, last_steering_angle, speed, throttle, info):

        if speed < 0.005:
            self.m_cnt_speed_zero += 1
            # print(self.m_cnt_speed_zero)

        blob_stats = self.m_twQTeamImageProcessor.trafficSignDetection(src_img)

        bIsSeenForkSign = False
        current_time    = float(info["elapsed"])
        # signInfo        = SimpleDetection.trafficSignDetection(src_img)

        # hard code
        if info["lap"] > self.m_current_lap:
            self.m_cnt_order = -1

        if len(blob_stats) >= 1 and current_time - self.m_time_traffic_sign > 1 and self.m_time_traffic_sign != 0:
            self.m_cnt_order += 1

        if len(blob_stats) >= 1:
            self.m_signInfo[0] = TRAFFIC_SIGN_ORDER[self.m_cnt_order]
            self.m_time_traffic_sign = current_time
        else:
            self.m_signInfo[0] = ''
        signInfo = self.m_signInfo

        if signInfo[0] == 'ForkLeft':
            # print('ForkLeft')
            self.m_cnt_fork_left_sign += 1
            self.m_last_fork_time = current_time
            bIsSeenForkSign = True
        elif signInfo[0] == 'ForkRight':
            # print('ForkRight')
            self.m_cnt_fork_right_sign += 1
            self.m_last_fork_time = current_time
            bIsSeenForkSign = True
        elif signInfo[0] == 'TurnLeft':
            # print('TurnLeft')
            self.m_cnt_turn_left_sign += 1
            self.m_last_turn_time = current_time
        elif signInfo[0] == 'TurnRight':
            # print('TurnRight')
            self.m_cnt_turn_right_sign += 1
            self.m_last_turn_time = current_time
        elif signInfo[0] == 'UTurnLeft':
            # print('UTurnLeft')
            self.m_cnt_u_turn_left_sign += 1
            self.m_last_u_turn_time = current_time
        elif signInfo[0] == 'UTurnRight':
            # print('UTurnRight')
            self.m_cnt_u_turn_right_sign += 1
            self.m_last_u_turn_time = current_time

        bIsForkLeftSign   = False
        bIsForkRightSign  = False
        bIsDetectForkSign = False
        if self.m_cnt_fork_left_sign >= 1:
            bIsForkLeftSign   = True
            bIsDetectForkSign = bIsForkLeftSign or bIsForkRightSign
            self.m_cnt_fork_left_sign = 0
        elif self.m_cnt_fork_right_sign >= 1:
            bIsForkRightSign  = True
            bIsDetectForkSign = bIsForkLeftSign or bIsForkRightSign
            self.m_cnt_fork_right_sign = 0

        if current_time - self.m_last_fork_time > 2 and (self.m_cnt_fork_left_sign > 0 or self.m_cnt_fork_right_sign > 0):
            self.m_cnt_fork_left_sign  = 0
            self.m_cnt_fork_right_sign = 0

        if current_time - self.m_last_turn_time > 2 and (self.m_cnt_turn_left_sign > 0 or self.m_cnt_turn_right_sign > 0):
            self.m_cnt_turn_left_sign = 0
            self.m_cnt_turn_right_sign = 0

        if current_time - self.m_last_u_turn_time > 2 and (self.m_cnt_u_turn_left_sign > 0 or self.m_cnt_u_turn_right_sign > 0):
            self.m_cnt_u_turn_left_sign = 0
            self.m_cnt_u_turn_right_sign = 0

        src_img  = cv2.resize(src_img, (160, 120), interpolation=cv2.INTER_LINEAR)
        proc_img = self.m_twQTeamImageProcessor.processImage(src_img)

        if self.m_is_in_fork_track == False:
            angle_index = 0
        else:
            angle_index = 2
            self.m_cnt_delay_frame += 1
        line_edges, line_rev, line_best_idx, slop_angle, slop_radian, steering_angle, offset_distance, offset_steering = self.getSteering(src_img, proc_img, angle_index)


        if line_best_idx == -1:
            if self.m_cnt_speed_zero >= 60:
                self.m_car.control(-40 * 0.01745, 0.1)
                return
        else:
            self.m_is_find_line = True
            self.m_cnt_speed_zero = 0


        if self.m_is_in_fork_track == False:
            output_steering = self.adjustSteeringInFullTrack(line_edges, line_best_idx, slop_angle, steering_angle, offset_distance, offset_steering)
        else:
            output_steering = self.adjustSteeringInForkTrack(line_edges, line_best_idx, slop_angle, steering_angle, offset_distance, offset_steering)

        if self.m_twQTeamImageProcessor.getTrackMode() == 0 or self.m_twQTeamImageProcessor.getTrackMode() == 1:
            if self.m_cnt_turn_left_sign >= 1 and line_best_idx <= 2:
                output_steering += self.m_const_radian_10
                self.m_cnt_turn_left_sign -= 1

            if self.m_cnt_turn_right_sign >= 1 and line_best_idx >= 4:
                output_steering -= self.m_const_radian_10
                self.m_cnt_turn_right_sign -= 1

        if self.m_is_in_fork_track == False:
            output_throttle = self.getThrottleInFullTrack()
        else:
            output_throttle = self.getThrottleInForkTrack()

        if bIsSeenForkSign:
            if speed > 0.3:
                output_throttle = -0.01

        bIsProcessForkSign= False
        if bIsForkLeftSign and self.m_is_in_fork_track == False:
            self.m_seen_fork_left_sign = True
        if self.m_seen_fork_left_sign:
            if line_best_idx > 3:
                output_steering -= self.m_const_radian_10
            elif line_best_idx == 3:
                output_steering -= self.m_const_radian_10
            elif line_best_idx == 2 and offset_distance > 0.5:
                output_steering -= self.m_const_radian_10
            else:
                self.m_is_in_fork_track    = True
                self.m_seen_fork_left_sign = False
                if self.m_twQTeamImageProcessor.m_initial_track == 0:
                    self.m_twQTeamImageProcessor.setTrackMode(track_mdoe=2)  # left fork mode
                else:
                    self.m_twQTeamImageProcessor.setTrackMode(track_mdoe=4)
            bIsProcessForkSign = True

        if bIsForkRightSign and self.m_is_in_fork_track == False:
            self.m_seen_fork_right_sign = True
        if self.m_seen_fork_right_sign:
            if line_best_idx < 3:
                output_steering += self.m_const_radian_10
            elif line_best_idx == 3:
                output_steering += self.m_const_radian_10
            elif line_best_idx == 4 and offset_distance < 0.5:
                output_steering += self.m_const_radian_10
            else:
                self.m_is_in_fork_track     = True
                self.m_seen_fork_right_sign = False
                if self.m_twQTeamImageProcessor.m_initial_track == 0:
                    self.m_twQTeamImageProcessor.setTrackMode(track_mdoe=3)  # right fork mode
                else:
                    self.m_twQTeamImageProcessor.setTrackMode(track_mdoe=5)
            bIsProcessForkSign = True

        if bIsProcessForkSign == False:

            if self.m_is_in_fork_track == False:
                output_steering, output_throttle = self.adjustControlBySharpTurnInFullTrack(line_best_idx, slop_angle, output_steering, output_throttle)
            else:
                output_steering, output_throttle = self.adjustControlBySharpTurnInForkTrack(line_best_idx, slop_angle, output_steering, output_throttle)

        if bIsDetectForkSign == False and self.m_is_in_fork_track and self.m_cnt_delay_frame >= 120:
            center_line_edges = self.m_twQTeamImageProcessor.doEdgeDetectionCenterLine(proc_img)
            bChangeTrackMode = False

            if len(center_line_edges) >= 3:
                bChangeTrackMode = True
            elif self.m_twQTeamImageProcessor.getTrackMode() == 2 and line_best_idx == 1:
                track_green_total = list(map(lambda x: len(x[x > 20]), [proc_img[1]]))
                if track_green_total[0] >= 20:
                    bChangeTrackMode = True
            elif self.m_twQTeamImageProcessor.getTrackMode() == 3 and line_best_idx == 2:
                track_red_total = list(map(lambda x: len(x[x > 20]), [proc_img[2]]))
                if track_red_total[0] >= 20:
                    bChangeTrackMode = True
            elif self.m_twQTeamImageProcessor.getTrackMode() == 4 and line_best_idx == 1:
                track_red_total = list(map(lambda x: len(x[x > 20]), [proc_img[2]]))
                if track_red_total[0] >= 20:
                    bChangeTrackMode = True
            elif self.m_twQTeamImageProcessor.getTrackMode() == 5 and line_best_idx == 2:
                track_green_total = list(map(lambda x: len(x[x > 20]), [proc_img[1]]))
                if track_green_total[0] >= 20:
                    bChangeTrackMode = True

            if bChangeTrackMode:
                # print('full track')
                self.m_is_in_fork_track  = False
                self.m_cnt_delay_frame   = 0
                if self.m_twQTeamImageProcessor.m_initial_track == 0:
                    self.m_twQTeamImageProcessor.setTrackMode(track_mdoe=0)
                else:
                    self.m_twQTeamImageProcessor.setTrackMode(track_mdoe=1)

        if info["lap"] > self.m_current_lap:
            self.m_current_lap = info["lap"]
            self.m_steering_pid.reset()
            self.m_distance_pid.reset()
            self.m_throttle_pid.reset()

        self.m_steering_history.append(output_steering)
        self.m_steering_history = self.m_steering_history[-self.m_max_steering_history:]
        self.m_throttle_history.append(output_throttle)
        self.m_throttle_history = self.m_throttle_history[-self.m_max_throttle_history:]
        self.m_distance_history.append(offset_distance)
        self.m_distance_history = self.m_distance_history[-self.m_max_distance_history:]
        send_steering = sum(self.m_steering_history)/self.m_max_steering_history
        send_throttle = sum(self.m_throttle_history)/self.m_max_throttle_history

        send_throttle = output_throttle
        if self.m_debug_mode:
            send_steering = 0.0
            send_throttle = 0.0
        self.m_car.control(send_steering, send_throttle)


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
        # brake               = float(dashboard["brakes"])
        speed               = float(dashboard["speed"])
        img                 = twQTeamImageProcessor.rgbTobgr(np.asarray(Image.open(BytesIO(base64.b64decode(dashboard["image"])))))

        # del dashboard["image"]
        # print('')
        # print(datetime.now(), dashboard)
        total_time = float(dashboard["time"])
        elapsed    = total_time
        # logit(elapsed)

        # if elapsed > 600:
        #     print("elapsed: " + str(elapsed))
        #     send_restart()

        info = {
            "lap"    : int(dashboard["lap"]) if "lap" in dashboard else 0,
            "elapsed": elapsed,
            "status" : int(dashboard["status"]) if "status" in dashboard else 0,
        }
        self._driver.on_dashboard(img, last_steering_angle, speed, throttle, info)


    def control(self, steering_angle, throttle):
        #convert the values with proper units
        steering_angle = min(max(twQTeamImageProcessor.radiusTodegree(steering_angle), -Car.MAX_STEERING_ANGLE), Car.MAX_STEERING_ANGLE)
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
    parser.add_argument(
        'record',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder to record the images.'
    )
    args = parser.parse_args()

    if args.record:
        if not os.path.exists(args.record):
            os.makedirs(args.record)

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
    drive = twQTeamDrive(car, args.record)

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

# vim: set sw=4 ts=4 et :
