import cv2
import numpy as np

img = cv2.imread('ceu.jpeg')
img2 = cv2.imread('sim.jpg')

cv2.imshow('sim', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

recorte = img2[700:1500, 500:1600]
cv2.imshow('pixels', recorte)
cv2.waitKey(0)
cv2.destroyAllWindows()