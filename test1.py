import glob
import pathlib

import cv2
import numpy as np

from fnc.normalize import normalize
from utils import show_image, inner_params, outer_params, get_circles, get_circle_blob


def process_image(im_path, out_path):
    pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)

    im = cv2.imread(im_path, 0)
    img = np.copy(im)

    circles = []

    c_prev, radius = get_circle_blob(im)

    cv2.circle(img, c_prev, radius, (0, 255, 0), 2)
    circles.append((*reversed(c_prev), radius))

    i = get_circles(im, *outer_params, c_prev)

    cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    circles.append((round(i[1]), round(i[0]), round(i[2])))

    polar_array, noise_array = normalize(im, circles[0][0], circles[0][1], circles[0][2],
                                         circles[1][0], circles[1][1], circles[1][2], radial_res=20, angular_res=240)

    polar_array = cv2.cvtColor(np.asarray(polar_array * 255, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    noise_array = cv2.cvtColor(np.asarray(noise_array * 255, dtype=np.uint8), cv2.COLOR_GRAY2BGR)

    cv2.imwrite(f"{out_path}/polar.png", polar_array)
    cv2.imwrite(f"{out_path}/noise.png", noise_array)
    cv2.imwrite(f"{out_path}/eye.png", img)


def main():
    files = glob.glob(r"data\CASIA-Iris-Twins\**\*.jpg", recursive=True)
    for i, file in enumerate(files):
        out = "/".join(file.replace(r"data\CASIA-Iris-Twins", r"data\output").split("\\"))
        process_image(file, out)

        print(f"{i + 1}/{len(files)} Processed")


if __name__ == "__main__":
    main()
