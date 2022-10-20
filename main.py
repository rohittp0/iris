import glob
import pathlib

import cv2
import numpy as np

from fnc.normalize import normalize
from fnc.segment import segment
from utils import show_image


def transform_image(img, threshold):
    retval, threshold = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

    open_close = cv2.bitwise_or(opening, closing, mask=None)

    return open_close


def process_image(im_path, out_path):
    im = cv2.imread(im_path, 0)

    ciriris, cirpupil, imwithnoise = segment(im, **params, eyelashes_thres=80)

    cv2.circle(im, [*reversed(ciriris[:2])], ciriris[2], (0, 255, 0), 2)
    cv2.circle(im, [*reversed(cirpupil[:2])], cirpupil[2], (255, 0, 255), 2)

    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    polar_array, noise_array = normalize(imwithnoise, ciriris[1], ciriris[0], ciriris[2],
                                         cirpupil[1], cirpupil[0], cirpupil[2], radial_res=20, angular_res=240)

    polar_array = cv2.cvtColor(np.asarray(polar_array * 255, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    noise_array = cv2.cvtColor(np.asarray(noise_array * 255, dtype=np.uint8), cv2.COLOR_GRAY2BGR)

    out_path = out_path.split(".")[0]

    cv2.imwrite(f"{out_path}_polar.png", polar_array)
    cv2.imwrite(f"{out_path}_noise.png", noise_array)
    cv2.imwrite(f"{out_path}_eye.png", im)

    np.savetxt(f"{out_path}_polar.txt", cv2.cvtColor(polar_array, cv2.COLOR_BGR2GRAY))
    np.savetxt(f"{out_path}_noise.txt", cv2.cvtColor(noise_array, cv2.COLOR_BGR2GRAY))
    np.savetxt(f"{out_path}_eye.txt", im)


def main():
    files = glob.glob(r"data\CASIA1\**\*.*", recursive=True)
    for i, file in enumerate(files):
        out = "/".join(file.replace(r"data\CASIA1", r"data\CASIA1_iris-full").split("\\"))
        process_image(file, out)

        print(f"{i + 1}/{len(files)} Processed")


if __name__ == "__main__":
    main()
