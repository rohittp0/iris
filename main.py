import glob
import pathlib

import cv2
import numpy as np

from fnc.normalize import normalize
from fnc.segment import segment


def process_image(im_path, out_path):
    pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)

    im = cv2.imread(im_path, 0)

    ciriris, cirpupil, imwithnoise = segment(im, eyelashes_thres=80)

    cv2.circle(im, [*reversed(ciriris[:2])], ciriris[2], (0, 255, 0), 2)
    cv2.circle(im, [*reversed(cirpupil[:2])], cirpupil[2], (255, 0, 255), 2)

    polar_array, noise_array = normalize(imwithnoise, ciriris[1], ciriris[0], ciriris[2],
                                         cirpupil[1], cirpupil[0], cirpupil[2], radial_res=20, angular_res=240)

    polar_array = cv2.cvtColor(np.asarray(polar_array * 255, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    noise_array = cv2.cvtColor(np.asarray(noise_array * 255, dtype=np.uint8), cv2.COLOR_GRAY2BGR)

    cv2.imshow("eye", im)
    cv2.waitKey()

    cv2.imwrite(f"{out_path}/polar.png", polar_array)
    cv2.imwrite(f"{out_path}/noise.png", noise_array)


def main():
    files = glob.glob(r"data\CASIA-Iris-Twins\**\*.jpg", recursive=True)
    for i, file in enumerate(files):
        out = "/".join(file.replace(r"data\CASIA-Iris-Twins", r"data\output").split("\\"))
        process_image(file, out)

        print(f"{i+1}/{len(files)} Processed")


if __name__ == "__main__":
    main()
