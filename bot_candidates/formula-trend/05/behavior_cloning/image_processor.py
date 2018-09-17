
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
        bottom_half_ratios = (0.55, 1.0)
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
            print("*** sharp turning")
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
        steering_angle = math.atan2(image_height, (px - camera_x))

        if debug:
            #draw the steering direction
            r = 60
            x = image_width / 2 + int(r * math.cos(steering_angle))
            y = image_height    - int(r * math.sin(steering_angle))
            cv2.line(img, (image_width / 2, image_height), (x, y), (255, 0, 255), 2)
            logit("steering angle: %0.2f, last steering angle: %0.2f" % (ImageProcessor.rad2deg(steering_angle), ImageProcessor.rad2deg(np.pi/2-last_steering_angle)))

        return (np.pi/2 - steering_angle) * 2.0
    
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
        #sobel_mask = mask_frame_sobel(img, kernel_size=9)
        #binary = np.logical_or(binary, sobel_mask)

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