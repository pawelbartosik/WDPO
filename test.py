import json
from pathlib import Path
from typing import Dict
import click
import cv2
from tqdm import tqdm
import numpy as np


# # HSV TRACKBAR
# def dummy(x):
#     pass
#
# img_path = "D:\\Studia\\Semestr 5\\WDPO Lab\\projekt\\WDPO\\data\\03.jpg"
# cv2.namedWindow('hsv')
# cv2.createTrackbar('Hmax', 'hsv', 0, 255, dummy)
# cv2.createTrackbar('Hmin', 'hsv', 0, 255, dummy)
# cv2.createTrackbar('Smax', 'hsv', 0, 255, dummy)
# cv2.createTrackbar('Smin', 'hsv', 0, 255, dummy)
# cv2.createTrackbar('Vmax', 'hsv', 0, 255, dummy)
# cv2.createTrackbar('Vmin', 'hsv', 0, 255, dummy)
# img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#
# while True:
#     key_code = cv2.waitKey(10)
#     if key_code == 27:
#         break
#
#     Hmax = cv2.getTrackbarPos('Hmax', 'hsv')
#     Hmin = cv2.getTrackbarPos('Hmin', 'hsv')
#     Smax = cv2.getTrackbarPos('Smax', 'hsv')
#     Smin = cv2.getTrackbarPos('Smin', 'hsv')
#     Vmax = cv2.getTrackbarPos('Vmax', 'hsv')
#     Vmin = cv2.getTrackbarPos('Vmin', 'hsv')
#
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     mask_hsv = cv2.inRange(hsv, (Hmin, Smin, Vmin), (Hmax, Smax, Vmax))
#     resize_img = cv2.resize(img, (500, 500))
#     resize_mask = cv2.resize(mask_hsv, (500, 500))
#
#     cv2.imshow('img', resize_img)
#     cv2.imshow('hsv_img', resize_mask)

# APPLY MASK
img_path = "D:\\Studia\\Semestr 5\\WDPO Lab\\projekt\\WDPO\\data\\03.jpg"
cv2.namedWindow('hsv')
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#TUTAJ MASKI


mask_hsv = cv2.inRange(hsv, (Hmin, Smin, Vmin), (Hmax, Smax, Vmax))
resize_img = cv2.resize(img, (500, 500))
resize_mask = cv2.resize(mask_hsv, (500, 500))

cv2.imshow('img', resize_img)
cv2.imshow('hsv_img', resize_mask)

