import cv2
import numpy as np

from utils import show_image

kernel = np.ones((5, 5), np.uint8)


def transform_image(img, threshold):
    retval, threshold = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

    open_close = cv2.bitwise_or(opening, closing, mask=None)

    return open_close


def segment(i):
    golden_reference = sum(sum(transform_image(i, 0)))
    working_img = None
    c = i.copy()

    for k in range(10, 1000, 10):

        working_img = transform_image(i, k)
        summing = sum(sum(working_img))
        difference = summing - golden_reference

        if difference > 800:
            break

    _, contours = cv2.findContours(working_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if contours is None:
        return

    for z in contours:

        x, y, w, h = cv2.boundingRect(z)
        if x + w < 150 and y + h < 200 and x - w // 4 > 0:
            cv2.rectangle(working_img, (x, y), (x + w, y + h), (0, 255, 0), -2)

    _, contours_2 = cv2.findContours(working_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    maximum_area = 0
    maximum_width = 0
    point_x = 0
    point_y = 0
    maximum_height = 0

    for z in contours_2:
        x, y, w, h = cv2.boundingRect(z)
        new_area = h * w
        if x + w < 150 and y + h < 200 and new_area > maximum_area and x - w // 4 > 0:
            maximum_area = new_area
            maximum_width = w
            point_x = x
            point_y = y
            maximum_height = h

    center_x = point_x + maximum_width // 2
    center_y = point_y + maximum_height // 2
    radius = 40

    if center_y - radius > 0 and center_x - radius > 0 and center_y + radius < 200 and center_x + radius < 150:
        new_roi = c[center_y - radius:center_y + radius, center_x - radius:center_x + radius]
        new_roi = cv2.resize(new_roi, (200, 150))
    else:
        center_y = c.shape[0] // 2
        center_x = c.shape[1] // 2
        new_roi = c[center_y - radius:center_y + radius, center_x - radius:center_x + radius]
        new_roi = cv2.resize(new_roi, (200, 150))

    show_image(new_roi, i)
