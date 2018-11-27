# -*- coding: utf-8 -*-
#
# Copyright 2018 isobar. All Rights Reserved.
#
# Usage:
#       python teeth-whitening.py pic.jpg
#

import sys
import cv2
import dlib
import numpy as np
from skimage import io
from PIL import Image


# brightness and contrast
# https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv/50053219#50053219
alpha = 1.0 # 1.0-3.0
beta = 50  # 0-100

predictor_path = "./data/shape_predictor_68_face_landmarks.dat"
CONST_IMAGE_PATH = "./faces/Tom_Cruise_avp_2014_4.jpg"
# CONST_IMAGE_PATH = "./faces/ko_p.jpg"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def alphaBlend(img1, img2, mask):
    """ alphaBlend img1 and img 2 (of CV_8UC3) with mask (CV_8UC1 or CV_8UC3)
    """
    if mask.ndim==3 and mask.shape[-1] == 3:
        alpha = mask/255.0
    else:
        alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)/255.0
    blended = cv2.convertScaleAbs(img1*(1-alpha) + img2*alpha)
    return blended
    
def shape2np(s):
    num = len(s.parts())
    np_points = np.zeros((num,2), np.int32)
    idx = 0
    for p in s.parts():
        np_points[idx] = (p.x, p.y)
        idx = idx + 1
    return np_points

def main():
    img = io.imread(image_path)
    faces = detector(img)

    if len(faces)==0:
        print("Face not found")
        return;

    for f, d in enumerate(faces):
        shape = predictor(img, d)
    
    np_points = shape2np(shape)

    # facial points
    # https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/

    # crop face
    crop_img = img[d.top():d.bottom(),d.left():d.right()]
    io.imsave("face.jpg", crop_img)
    
    # crop mouth
    mouth_max_point = np.max(np_points[60:], axis=0)
    mouth_min_point = np.min(np_points[60:], axis=0)
    io.imsave("mouth.jpg", img[mouth_min_point[1]:mouth_max_point[1], mouth_min_point[0]:mouth_max_point[0]])

    # mouth: 48-67
    # teeth: 60-67
    # create blank image
    mask = np.zeros((d.bottom()-d.top(), d.right()-d.left()), np.uint8)

    # load and save image
    # im = Image.fromarray(mask)
    # im.save("blank.jpg", im)

    # create teeth mask
    cv2.fillConvexPoly(mask, np_points[60:]-(d.left(), d.top()), 1)
    cv2.imwrite("mask.jpg", mask)
    crop_jpg_with_mask= cv2.bitwise_and(crop_img, crop_img, mask = mask)

    # smoothing mask
    blur_mask = cv2.GaussianBlur(crop_jpg_with_mask,(21,21), 11.0)
    io.imsave('blur_mask.jpg', blur_mask)
    
    # convert rgb2rgba
    crop_png = cv2.cvtColor(crop_img, cv2.COLOR_RGB2RGBA)
    np_alpha = blur_mask[:, :, 0]/255.0
    crop_png[:, :, 3] = blur_mask[:, :, 0]
    
    # brightness and contrast
    # Ref: https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
    crop_png_with_brightness = cv2.convertScaleAbs(crop_png, alpha=alpha, beta=beta)
    #crop_png_with_brightness = np.zeros(crop_png.shape, crop_png.dtype)
    #for y in range(crop_png.shape[0]):
    #    for x in range(crop_png.shape[1]):
    #        b,g,r,c = crop_png[y,x]
    #        if (c!=0):
    #            crop_png_with_brightness[y,x] = np.clip(alpha*crop_png[y,x] + beta, 0, 255)

    io.imsave("brightness.png", crop_png_with_brightness)
    # output
    output = np.zeros(crop_img.shape, crop_img.dtype)
    # merge two images with alpha channel
    # Ref: https://stackoverflow.com/questions/41508458/python-opencv-overlay-an-image-with-transparency
    output[:, :, 0] = (1.0 - np_alpha) * crop_png[:, :, 0] + np_alpha * crop_png_with_brightness[:, :, 0]
    output[:, :, 1] = (1.0 - np_alpha) * crop_png[:, :, 1] + np_alpha * crop_png_with_brightness[:, :, 1]
    output[:, :, 2] = (1.0 - np_alpha) * crop_png[:, :, 2] + np_alpha * crop_png_with_brightness[:, :, 2]
    io.imsave("output.jpg", output)

    #crop_png = cv2.add(src_png, crop_png_with_brightness)
    # cv2.imwrite("output.jpg", output)

if __name__ == "__main__":
    global image_path
    if (len(sys.argv)>1):
        image_path = sys.argv[1]
    else:
        image_path = CONST_IMAGE_PATH
    main()
