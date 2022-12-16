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
# img_path = "D:\\Studia\\Semestr 5\\WDPO Lab\\projekt\\WDPO\\data\\05.jpg"
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

#DO POPRAWKI
#05 czerwony
#06 dosyc male pole
#07 czerwony ale nie tak zle
#11 fioletowy not bad, 12 to samo
#21, 23, 24, 25 i kolejne fioletowy
#30 fioletowy
#31 fioletowy
# TROCHE CIEMNIEJSZY FIOLETOWY I CZEWRWONY, ZOBACZYC JAK Z ZMNIEJSZANIE I POTEM SPRAWDZANIE POLA

# APPLY MASK
img_path = "D:\\Studia\\Semestr 5\\WDPO Lab\\projekt\\WDPO\\data\\39.jpg"
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
blur_bilateral = cv2.bilateralFilter(img, 15, 75, 75)
hsv = cv2.cvtColor(blur_bilateral, cv2.COLOR_BGR2HSV)
# blur_median = cv2.medianBlur(img, 10)
# cv2.imshow('blur', blur_bilateral)
#TUTAJ MASKI
mask_hsv_green = cv2.inRange(hsv, (33, 190, 0), (146, 255, 255))
mask_hsv_purple = cv2.inRange(hsv, (138, 84, 0), (170, 255, 255))
mask_hsv_yellow = cv2.inRange(hsv, (6, 203, 83), (30, 255, 255))
mask_hsv_red = cv2.inRange(hsv, (170, 43, 125), (255, 255, 255))

#mask_hsv = cv2.inRange(hsv, (Hmin, Smin, Vmin), (Hmax, Smax, Vmax))
kernel = np.ones((10, 10), np.uint8)
kernel1 = np.ones((9, 9), np.uint8) #11, 11 green,
kernel2 = np.ones((11, 11), np.uint8)#5, 5 green,
#erosion = cv2.erode(mask_hsv_purple, kernel, iterations=1)
opening = cv2.morphologyEx(mask_hsv_green, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel1)
erosion = cv2.erode(closing, kernel1, iterations=1)
dilation = cv2.dilate(erosion, kernel2, iterations=1)
resize_closing = cv2.resize(closing, (500, 500))
resize_opening = cv2.resize(opening, (500, 500))
dilation_res = cv2.resize(dilation, (500, 500))

#
white = np.argwhere(dilation == 255)
# array = white[0]
# print(white)
# print(white[10, 0])
# print(white[10, 1])
# items = []
# for i in white:
#     print(i[0])

cnt, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(len(cnt))

resize_img = cv2.resize(img, (500, 500))
resize_mask_green = cv2.resize(mask_hsv_green, (500, 500))
resize_mask_purple = cv2.resize(mask_hsv_purple, (500, 500))
resize_mask_yellow = cv2.resize(mask_hsv_yellow, (500, 500))
resize_mask_red = cv2.resize(mask_hsv_red, (500, 500))

cv2.imshow('img', resize_img)
cv2.imshow('green', resize_mask_green)
# cv2.imshow('purple', resize_mask_purple)
# cv2.imshow('yellow', resize_mask_yellow)
# cv2.imshow('red', resize_mask_red)
#
# cv2.imshow('closing', resize_closing)
cv2.imshow('opening', resize_opening)
cv2.imshow('dilation', dilation_res)
cv2.waitKey()



