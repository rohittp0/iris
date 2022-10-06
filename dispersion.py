import cv2
from sklearn.preprocessing import normalize
from scipy.special import entr
import matplotlib.pyplot as plt

polar = cv2.imread("data/output/02/1R/S3021R03.jpg/polar.png", cv2.IMREAD_GRAYSCALE)

polar_normal_row = normalize(polar)
polar_normal_col = normalize(polar, axis=0)

dispersion_rows = entr(polar_normal_row).sum(axis=1)
dispersion_cols = entr(polar_normal_col).sum(axis=0)

for title, data in [("Row", dispersion_rows), ("Column", dispersion_cols)]:
    figure, axis = plt.subplots()
    axis.set_title("Dispersion along "+title)
    axis.set_xlabel(title)
    axis.plot(data)

    figure.savefig("data/" + title)

# tau =1
# m = 4, 5, 6
# c = 3, 4 ,5
# logx = e

# SpecEn
# N = 512
# freq = none
