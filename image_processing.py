import cv2
import numpy as np

from math import floor


PROCESSED_IMAGE_SHAPE = (64, 200, 3)
HOOD_PART = 0.15
SKY_PART = 0.25


def get_processed_image_shape():
    return PROCESSED_IMAGE_SHAPE

def crop_hood(img):
    return img[:-floor(img.shape[0]*HOOD_PART),:,:]

def crop_sky(img):
    return img[floor(img.shape[0]*SKY_PART):,:,:]

def normalize_image(in_img):
    out_img = np.empty_like(in_img)
    cv2.normalize(in_img, out_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return out_img

def process_image(img):
    img = crop_hood(img)
    img = crop_sky(img)
    img = resize_image(img)
    return img
