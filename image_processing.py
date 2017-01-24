import cv2
import numpy as np

from math import floor

HOOD_HEIGHT = 24
SCALE = 0.5

def get_processed_image_shape(in_img):
    return floor((in_img.shape[0]-HOOD_HEIGHT) * SCALE), floor(in_img.shape[1] * SCALE), in_img.shape[2]

def process_image(in_img):
    crop_img = in_img[:-HOOD_HEIGHT, :, :]
    yuv_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2YCrCb)
    norm_img = np.zeros_like(yuv_img)
    cv2.normalize(yuv_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    resized_img = cv2.resize(norm_img, (0,0), fx=SCALE, fy=SCALE)
    return resized_img
