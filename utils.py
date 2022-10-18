import math

import cv2
import numpy as np


def show_image(*image):
    for i, img in enumerate(image):
        cv2.imshow(f"image {i}", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


inner_params = (50, 60, 30, 40)
outer_params = (50, 50, 70, 180)


def find_iris(im, thresh):
    _, binaryIm = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

    return cv2.dilate(binaryIm, kernel, 1)


def get_circles(image, param1, param2, min_r, max_r, c_prev=None):
    img = cv2.medianBlur(image, 5)

    circles = None

    for i in range(48):
        for j in range(2):
            if j % 2:
                param1 -= 1
            else:
                param2 -= 1

            circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 10, param1=param1,
                                       param2=param2, minRadius=min_r, maxRadius=max_r)

            if circles is not None:
                break

    circles = np.uint16(np.around(circles[0, :]))[:]
    ret = circles[0]

    if len(circles) > 1:
        distance = 5675675675670
        for circle in circles:
            center = circle[0], circle[1]
            if c_prev is not None:
                d = math.dist(center, c_prev)
            else:
                d = math.dist(center, (image.shape[1] / 2, image.shape[0] / 2))

            if d < distance:
                distance = d
                ret = circle

    return ret


def get_circle_blob(image):
    im = cv2.GaussianBlur(image, (11, 11), 0)
    im = cv2.inRange(im, 23, 30)
    ret, im = cv2.threshold(im, 127, 255, 0)

    contours, _ = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnt = max(contours, key=cv2.contourArea)

    (x, y), radius = cv2.minEnclosingCircle(cnt)

    center = (int(x), int(y))
    radius = int(radius)

    return center, radius
