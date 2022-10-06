import math

import cv2
import numpy as np


def show_image(*image):
    for i, img in enumerate(image):
        cv2.imshow(f"image {i}", img)
    cv2.waitKey()


inner_params = (50, 60, 30, 40)
outer_params = (50, 50, 70, 180)


def find_eyelid(image, iris_center, min_r, max_r):
    row_min, row_max = iris_center[1] + min_r, min(iris_center[1] + max_r, image.shape[0])
    col_min, col_max = max(iris_center[0] - max_r, 0), min(iris_center[0] + max_r, image.shape[1])
    row_min = iris_center[1] if row_min >= image.shape[0] else row_min

    image = image[row_min:row_max, col_min:col_max]
    image = cv2.GaussianBlur(image, (11, 11), 0)
    image = cv2.inRange(image, 50, 90)

    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    cont, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cont:
        print("Couldn't detect eyelid", end=" ")
        return None, None

    cont = max(cont, key=cv2.contourArea)

    cont = cont[:, 0, :]
    cent = np.tile(iris_center, (cont.shape[0], 1))
    cont = cont + np.tile((col_min, row_min), (cont.shape[0], 1))

    dist = cont - cent
    dist = np.sum(dist * dist, axis=1)

    cont = cont[dist > math.pow(min_r, 2)]
    dist = dist[dist > math.pow(min_r, 2)]

    if len(dist) == 0:
        print("All points too close to iris", end=" ")
        return None, None

    index = np.argmin(dist)

    return round(math.sqrt(dist[index])) - 2, cont[index]


def get_circles(image, r_prev, c_prev=None):
    c_prev = c_prev or (image.shape[1] / 2, image.shape[0] / 2)
    r_min, r_max = round(r_prev * 1.8), round(r_prev * 4)

    lid, point = find_eyelid(np.copy(image), c_prev, r_min, r_max)

    if lid is not None and point is not None:
        return (*c_prev, lid), point

    img = cv2.medianBlur(image, 5)
    param1, param2 = 50, 50

    all_c = []

    for i in range(48):
        for j in range(2):
            if j % 2:
                param1 -= 1
            else:
                param2 -= 1

            circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 10, param1=param1,
                                       param2=param2, minRadius=r_min, maxRadius=r_max)

            if circles is not None:
                all_c.append(np.uint16(np.around(circles[0, :]))[:])

    ret = None
    distance = 5675675675670

    for circle_set in all_c:
        for circle in circle_set:
            center = circle[0], circle[1]
            if c_prev is not None:
                d = math.dist(center, c_prev)
            else:
                d = math.dist(center, (image.shape[1] / 2, image.shape[0] / 2))

            if d < distance:
                distance = d
                ret = circle

    return ret, (0, 0)


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
