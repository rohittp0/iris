#               p1  p2  rmi rma

# eye = cv2.imread('data/CASIA-Iris-Twins/00/1L/S3001L02.jpg', 0)
#
# # for i in range(100):
# #     cv2.imshow(f"{i}", cv2.threshold(eye, i, 255, cv2.THRESH_BINARY)[1])
# #     cv2.waitKey()
#
# # eye = cv2.threshold(eye, 50, 255, cv2.THRESH_BINARY)[1]
# img = cv2.cvtColor(eye, cv2.COLOR_GRAY2BGR)


# c_prev = None
# circles = []
#
# for c in [inner_params, outer_params]:
#     i = get_circles(eye, *c, c_prev)
#
#     c_prev = (i[0], i[1])
#     # draw the outer circle
#     cv2.circle(img, c_prev, i[2], (0, 255, 0), 2)
#     # draw the center of the circle
#     cv2.circle(img, c_prev, 2, (0, 0, 255), 3)
#     show_image(img)
#
