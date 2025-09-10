import cv2
import numpy as np

img = cv2.imread('images.jpeg')
img2 = cv2.imread('belo.jpeg')

# cv2.imshow('sim', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img1 = cv2.imread('belo.jpeg', cv2.IMREAD_GRAYSCALE)
# cv2.imshow('cinza niveis', img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite('cinza.jpeg', img1)

# print(img.shape)
# print(img.size)
# print(img.dtype)
# print('[B G R]: {}'.format(img[50, 50]))
# img[0:20, 0:200] = [2, 102, 202]
# img[0, 200] = [0, 0, 0]
# cv2.imshow('pixels', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

recorte = img[5:10, 5:10]
img[0:5, 0:5] = recorte
cv2.imshow('pixels', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print('Imagem 1: {} {}'.format(img.shape, img.dtype))
print('Imagem 2: {} {}'.format(img2.shape, img2.dtype))

img3 = img*0.5 + img2*0.5
img3 = img3.astype(np.uint8)
print('Imagem 3: {} {}'.format(img3.shape, img3.dtype))

cv2.imshow('soma', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()