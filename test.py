import glob

import cv2
import numpy as np

from utils import show_image


def main():
    files = glob.glob(r"data\IITD Database\**\*.*", recursive=True)
    with open("data/output1/README.md", "w") as readme:
        for i, file in enumerate(files):
            out = "/".join(file.replace(r"data\IITD Database", r"data\output1").split("\\"))
            out = f"![eye image](./{out}/eye.png)\n"

            print(out)
            readme.write(out)


i = cv2.imread("data/output/02/1R/S3021R03.jpg/polar.png", 0)

i[:, 20:100] = 0
i[:, 140:220] = 0

j = np.zeros((20, 60))

j[:, :20] = i[:, :20]
j[:, 20:40] = i[:, 100:120]
j[:, 40:60] = i[:, 220:]

cv2.imwrite("roi.png", j)
# if __name__ == "__main__":
#     main()
