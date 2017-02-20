import numpy as np
import cv2
import matplotlib.pyplot as plot
from sklearn.externals import joblib
from skimage.feature import hog


def rgb_to_gray_line(img_rgb):
    img_gray = np.ndarray((img_rgb.shape[0], img_rgb.shape[1]))
    for i in np.arange(0, img_rgb.shape[0]):
        for j in np.arange(0, img_rgb.shape[1]):
            if img_rgb[i, j, 2] < 50 and img_rgb[i, j, 1] < 50 and not img_rgb[i, j, 0] < 150:
                img_gray[i, j] = 255
            else:
                img_gray[i, j] = 0
    img_gray = img_gray.astype('uint8')
    img_gray_dl = cv2.dilate(img_gray, cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6)), iterations=2)

    return img_gray_dl


def rgb_to_gray_number(img_rgb):
    img_gray = np.ndarray((img_rgb.shape[0], img_rgb.shape[1]))
    whiteness = 154
    for i in np.arange(0, img_rgb.shape[0]):
        for j in np.arange(0, img_rgb.shape[1]):
            if img_rgb[i, j, 0] > whiteness and img_rgb[i, j, 1] > whiteness and img_rgb[i, j, 2] > whiteness:
                img_gray[i, j] = 255
            else:
                img_gray[i, j] = 0
    img_gray = img_gray.astype('uint8')
    img_gray_dl = cv2.dilate(img_gray, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=2)

    return img_gray_dl


def two_point_line_equation(x1, y1, x2, y2):
    y22 = y2+0.000
    y11 = y1+0.000
    k = (y22-y11)/(x2-x1)
    n = k * (-x1) + y1

    return [k, n]


def line_region_overlap(k, n, x, y):
    yy = k * x + n
    if abs(yy-y) < 1:
        return True
    else:
        return False


clf = joblib.load("digits_cls.pkl")
file = open('outx.txt', 'w')
file.close()
final_sum = 0

for p in np.arange(0, 1):

    cap = cv2.VideoCapture("Videos/video-" + str(p) + ".avi")
    file = open('Log2.txt', 'w')
    file.close()
    x = 0
    cap.set(1, x)
    while True:
        x = x + 1
        ret, frame = cap.read()
        if not ret:
            break

        k11 = 0
        n11 = 0
        k0 = 0
        n0 = 0
        k22 = 0
        n22 = 0
        k2 = 0
        n2 = 0

        sum = []

        im_line = rgb_to_gray_line(frame)
        _, ctrs_line, _ = cv2.findContours(im_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects_line = [cv2.boundingRect(ctr) for ctr in ctrs_line]
        rect_line = rects_line[0]
        h_line = rect_line[3]
        w_line = rect_line[2]

        [k0, n0] = two_point_line_equation(rect_line[0], 480 - rect_line[1] - rect_line[3], rect_line[0] + rect_line[2], 480 - rect_line[1])
        ww = (rect_line[0], rect_line[0] + rect_line[2])

        cv2.rectangle(frame, (rect_line[0], rect_line[1]), (rect_line[0] + rect_line[2], rect_line[1] + rect_line[3]), (255, 0, 0), 2)

        im_number = rgb_to_gray_number(frame)
        _, ctrs_number, _ = cv2.findContours(im_number.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects_number = [cv2.boundingRect(ctr) for ctr in ctrs_number]

        for rect in rects_number:
            h = rect[3]
            w = rect[2]

            x_c = rect[0] + rect[2] / 2
            y_c = 480 - (rect[1] + rect[3] / 2)

            rect_center = (x_c, y_c)

            leng = int(rect[3] * 1.6)
            pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
            pt2 = int(rect[0] + rect[2] // 2 - leng // 2)

            pt1 = 0 if pt1 < 0 else pt1
            pt2 = 0 if pt2 < 0 else pt2

            roi = im_number[pt1:pt1 + leng, pt2:pt2 + leng]

            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            roi = cv2.dilate(roi, (3, 3))

            roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
            nbr = clf.predict(np.array([roi_hog_fd], 'float64'))

            if ww[0] < x_c < ww[1] and line_region_overlap(k0, n0, x_c, y_c):
                cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)
                final_sum += int(nbr)

                file = open('Log2.txt', 'a')
                file.write('suma: ' + str(final_sum) + ' -- frame: ' + str(x) +  ' otprilike je: ' + str(x / 40.0) + ' sec\n')
                file.write('x_c: ' + str(x_c) + 'y_c: ' + str(y_c) + '\n')
                file.close()
            else:
                cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 2)

            sum.append(int(nbr))
            cv2.putText(frame, str(int(nbr[0])), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

        print 'suma: ' + str(final_sum) + ' -- frame: ' + str(x) +  ' otprilike je: ' + str(x / 40.0) + ' sec'

    file = open('out.txt', 'a')
    file.write("Videos/video-" + str(p) + ".avi" + '\t' + str(final_sum) + '\n')
    file.close()
    cap.release()
