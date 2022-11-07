import glob
import pathlib

import cv2
import numpy as np

from fnc.normalize import normalize
from fnc.segment import segment
from utils import show_image


def process_image(im_path, out_path):
    read = cv2.imread(im_path, 0)
    params = {"minrad": 10, "multiplier": 0.8, "sigma": 2, "virt": 0, "horiz": 1}

    while True:
        im = read.copy()

        ciriris, cirpupil, imwithnoise = segment(im, **params, eyelashes_thres=80)

        cv2.circle(im, [*reversed(ciriris[:2])], ciriris[2], (0, 255, 0), 2)
        cv2.circle(im, [*reversed(cirpupil[:2])], cirpupil[2], (255, 0, 255), 2)

        show_image(im)

        choice = int(input("1) Save\n 2) Retry\n 3) Skip\n"))

        if choice == 2:
            for param in params:
                params[param] = input(f"{param} ({params[param]}): ")
                if params[param].isdigit():
                    params[param] = int(params[param])
                else:
                    params[param] = float(params[param])
        elif choice == 1:
            break
        elif choice == 3:
            return

    out = pathlib.Path(out_path)

    pathlib.Path(out.parent).parent.mkdir(parents=True, exist_ok=True)

    polar_array, noise_array = normalize(imwithnoise, ciriris[1], ciriris[0], ciriris[2],
                                         cirpupil[1], cirpupil[0], cirpupil[2], radial_res=20, angular_res=240)

    polar_array = cv2.cvtColor(np.asarray(polar_array * 255, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    noise_array = cv2.cvtColor(np.asarray(noise_array * 255, dtype=np.uint8), cv2.COLOR_GRAY2BGR)

    cv2.imwrite(f"{out.name}_polar.png", polar_array)
    cv2.imwrite(f"{out.name}_noise.png", noise_array)
    cv2.imwrite(f"{out.name}_eye.png", im)

    np.savetxt(f"{out.name}_polar.txt", cv2.cvtColor(polar_array, cv2.COLOR_BGR2GRAY))
    np.savetxt(f"{out.name}_noise.txt", cv2.cvtColor(noise_array, cv2.COLOR_BGR2GRAY))
    np.savetxt(f"{out.name}_eye.txt", im)


def main():
    data_folder = input("Enter the data folder : ")
    out_base = f"data/{data_folder}_iris_full/"

    files = glob.glob(f"data/{data_folder}/**/*.*", recursive=True)
    for i, file in enumerate(files):
        # if i < 3170:
        #     continue

        out = "/".join(file.replace(f"data/{data_folder}", out_base).split("\\"))
        ext = "_".join([x.split(".")[0] for x in out.split(out_base)[-1].split("/")])[1:]

        out = f"{out_base}{ext.split('_')[0]}/{ext}"

        process_image(file, out)

        print(f"{i + 1}/{len(files)} Processed")


if __name__ == "__main__":
    main()
