# import cv2
# img = cv2.imread("E:\\ctpn_yi\\dataset\\for_train\\Imageset\\T1.AK_XX8hXXbnu_Z1_042512.jpg.jpg")
# color = (255, 255, 255)
# cv2.line(img, (0, 400), (300, 400), color, thickness=11)
# cv2.imshow("yi", img)
# cv2.waitKey(0)

# import numpy as np
# a = np.array([[1, 2],[3, 4], [5, 6]])
# b = tuple(a)
# print(b)
#
# c = [(row[0], row[1]) for row in a]
# print(tuple(c))
# print("in file {}, x2 must be larger than x1 in anchor".format(__file__))

import numpy as np
a = np.logspace(start=0, stop=16, num=16, base=1.25, endpoint=False)*8
print(a)
