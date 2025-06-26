# scripts/rot_segmentation.py

import cv2
import numpy as np

def estimate_rot_percent(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    rot_pixels = cv2.countNonZero(mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    percent = (rot_pixels / total_pixels) * 100
    return round(percent, 2)