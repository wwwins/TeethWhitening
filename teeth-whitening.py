# -*- coding: utf-8 -*-
#
# Copyright 2018 isobar. All Rights Reserved.
#
# Usage:
#       python teeth-whitening.py pic.jpg
#

import os
import sys
import argparse
import cv2
import dlib
import numpy as np
from skimage import io
from PIL import Image
from scipy.spatial import distance
import IsobarImg

MAR = 0.30
# 100*100
FACE_IMAGE_SIZE = 10000

# brightness and contrast
# https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv/50053219#50053219

#CONST_PREDICTOR_PATH = "./data/shape_predictor_68_face_landmarks.dat"
CONST_IMAGE_PATH = "./faces/Tom_Cruise_avp_2014_4.jpg"
# CONST_IMAGE_PATH = "./faces/ko_p.jpg"

parser = argparse.ArgumentParser(description='teeth whitening editor')
parser.add_argument('predictor_path', help='predictor file')
parser.add_argument('file', help='image file')
parser.add_argument('-a', metavar='alpha', default='1.0', type=float, help='alpha value range: 1.0-3.0')
parser.add_argument('-b', metavar='beta', default='50', type=int, help='beta value range: 0-100')
args = parser.parse_args()

alpha = args.a
beta = args.b
#alpha = 1.0 # 1.0-3.0
#beta = 50  # 0-100

predictor_path = args.predictor_path

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# compute the mouth aspect ratio
# opening mouth: mar > 0.30
def mouth_aspect_ratio(mouth):
    D = distance.euclidean(mouth[33], mouth[51])
    # D1 = distance.euclidean(mouth[50], mouth[58])
    # D2 = distance.euclidean(mouth[51], mouth[57])
    # D3 = distance.euclidean(mouth[52], mouth[56])
    D1 = distance.euclidean(mouth[61], mouth[67])
    D2 = distance.euclidean(mouth[62], mouth[66])
    D3 = distance.euclidean(mouth[63], mouth[65])
    mar = (D1+D2+D3)/(3*D)
    print("mar={}".format(mar))
    return mar;
    
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
    filename = os.path.basename(image_path)
    publicname = os.path.dirname(image_path)[:-6]
    dirname = os.path.join(publicname, 'result', filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    img = io.imread(image_path)
    h,w,c = img.shape
    if c == 4:
        img = img[:,:,:3]
    io.imsave(dirname+"/before.jpg", img)
    res = IsobarImg.beautifyImage(dirname+"/before.jpg")
    res.save(dirname+"/after.jpg")

    faces = detector(img, 1)

    if len(faces)==0:
        print("Face not found")
        return

    max_face = 0
    max_face_id = 0
    for f, d in enumerate(faces):
        face_box = (d.bottom()-d.top())*(d.right()-d.left())
        if face_box > max_face:
            max_face = face_box
            max_face_id = f
    
    for f, d in enumerate(faces):
        if f == max_face_id:
            shape = predictor(img, d)
            break
    
    if (d.bottom()-d.top())*(d.right()-d.left()) < FACE_IMAGE_SIZE:
        print("Face too small:{}".format(max_face))
        return

    np_points = shape2np(shape)

    # detect an open mouth
    if (mouth_aspect_ratio(np_points)<MAR):
        print("Mouth not open")
        return

    # facial points
    # https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/

    # crop face
    crop_img = img[d.top():d.bottom(),d.left():d.right()]
    if crop_img.size == 0:
        return
    io.imsave(dirname+"/face.jpg", crop_img)
    
    # crop mouth
    mouth_max_point = np.max(np_points[60:], axis=0)
    mouth_min_point = np.min(np_points[60:], axis=0)
    io.imsave(dirname+"/mouth.jpg", img[mouth_min_point[1]:mouth_max_point[1], mouth_min_point[0]:mouth_max_point[0]])

    # mouth: 48-67
    # teeth: 60-67
    # create blank image
    mask = np.zeros((d.bottom()-d.top(), d.right()-d.left()), np.uint8)

    # load and save image
    # im = Image.fromarray(mask)
    # im.save("blank.jpg", im)

    # create teeth mask
    cv2.fillConvexPoly(mask, np_points[60:]-(d.left(), d.top()), 1)
    cv2.imwrite(dirname+"/mask.jpg", mask)
    crop_jpg_with_mask= cv2.bitwise_and(crop_img, crop_img, mask = mask)

    # smoothing mask
    blur_mask = cv2.GaussianBlur(crop_jpg_with_mask,(21,21), 11.0)
    io.imsave(dirname+'/blur_mask.jpg', blur_mask)
    
    # convert rgb2rgba
    crop_png = cv2.cvtColor(crop_img, cv2.COLOR_RGB2RGBA)
    np_alpha = blur_mask[:, :, 0]/255.0
    crop_png[:, :, 3] = blur_mask[:, :, 0]
    
    # brightness and contrast
    # Ref: https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
    # cv2.convertScaleAbs is more faster
    crop_png_with_brightness = cv2.convertScaleAbs(crop_png, alpha=alpha, beta=beta)
    #crop_png_with_brightness = np.zeros(crop_png.shape, crop_png.dtype)
    #for y in range(crop_png.shape[0]):
    #    for x in range(crop_png.shape[1]):
    #        b,g,r,c = crop_png[y,x]
    #        if (c!=0):
    #            crop_png_with_brightness[y,x] = np.clip(alpha*crop_png[y,x] + beta, 0, 255)

    io.imsave(dirname+"/brightness.png", crop_png_with_brightness)
    # output
    output = np.zeros(crop_img.shape, crop_img.dtype)
    # merge two images with alpha channel
    # Ref: https://stackoverflow.com/questions/41508458/python-opencv-overlay-an-image-with-transparency
    #output[:, :, 0] = (1.0 - np_alpha) * crop_png[:, :, 0] + np_alpha * crop_png_with_brightness[:, :, 0]
    #output[:, :, 1] = (1.0 - np_alpha) * crop_png[:, :, 1] + np_alpha * crop_png_with_brightness[:, :, 1]
    #output[:, :, 2] = (1.0 - np_alpha) * crop_png[:, :, 2] + np_alpha * crop_png_with_brightness[:, :, 2]
    np_alpha = np_alpha.reshape(crop_img.shape[0], crop_img.shape[1], 1)
    output[:, :, :] = (1.0 - np_alpha) * crop_png[:, :, :3] + np_alpha * crop_png_with_brightness[:, :, :3]
    io.imsave(dirname+"/output.jpg", output)

    #crop_png = cv2.add(src_png, crop_png_with_brightness)
    # cv2.imwrite("output.jpg", output)

    # save before and after images
    # io.imsave(dirname+"/before.jpg", img)
    img[d.top():d.bottom(),d.left():d.right()] = output
    io.imsave(dirname+"/whitening.jpg", img)
    res = IsobarImg.beautifyImage(dirname+"/whitening.jpg")
    res.save(dirname+"/after.jpg")

if __name__ == "__main__":
    global image_path
    if (len(sys.argv)>1):
        image_path = sys.argv[2]
    else:
        image_path = CONST_IMAGE_PATH
    main()
