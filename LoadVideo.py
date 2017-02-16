# Load video, grab frame and apply morphological transformations
import cv2
import matplotlib.pyplot as plot
import numpy as np

video = cv2.VideoCapture("../Videos/video-0.avi")
ret_val, frame = video.read()

if ret_val:
    plot.imshow(frame)
    plot.show()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plot.imshow(gray_frame, 'gray')
    plot.show()

    ret_val_th, threshold_img = cv2.threshold(gray_frame, 120, 255, cv2.THRESH_BINARY)
    plot.imshow(threshold_img, 'gray')
    plot.show()

    kernel = np.ones((2, 2), np.uint8)
    open_img = cv2.morphologyEx(threshold_img, cv2.MORPH_OPEN, kernel)
    plot.imshow(open_img, 'gray')
    plot.show()
