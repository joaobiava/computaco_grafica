import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

img = cv2.imread('imagens/lena.png')
img2 = cv2.imread('imagens/goblin_machine.jpeg')

img1_pb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
img2_pb = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) 

# cv2.imshow('sim', img1_pb)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('sim', img2_pb)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def image_convolution(f, w, debug=False):
    N,M = f.shape
    n,m = w.shape
    
    a = int((n-1)/2)
    b = int((m-1)/2)

    # obtem filtro invertido
    w_flip = np.flip( np.flip(w, 0) , 1)

    g = np.zeros(f.shape, dtype=np.uint8)

    # para cada pixel:
    for x in range(a,N-a):
        for y in range(b,M-b):
            # obtem submatriz a ser usada na convolucao
            sub_f = f[ x-a : x+a+1 , y-b:y+b+1 ]
            if (debug==True):
                print(str(x)+","+str(y)+" - subimage:\n"+str(sub_f))
            # calcula g em x,y
            g[x,y] = np.sum( np.multiply(sub_f, w_flip)).astype(np.uint8)

    return g

w_med = np.matrix([[1, 1, 1], [1, 1, 1], [1, 1, 1]])/9.0

img2_media = image_convolution(img2_pb, w_med)

# exibindo imagem original e filtrada por w_med
plt.figure(figsize=(12,12)) 
plt.subplot(121)
plt.imshow(cv2.cvtColor(img1_pb, cv2.COLOR_BGR2RGB), cmap="gray", vmin=0, vmax=255)
plt.title("imagem original, ruidosa")
plt.axis('off')
plt.subplot(122)
plt.imshow(cv2.cvtColor(img2_media, cv2.COLOR_BGR2RGB), cmap="gray", vmin=0, vmax=255)
plt.title("imagem convoluída com filtro de media")
plt.axis('off')
plt.show()

# w_diff = np.matrix([[ 0, -1,  0], 
#                     [-1,  4, -1], 
#                     [ 0, -1,  0]])
# print(w_diff)

# img1_diff = image_convolution(img1_pb, w_diff)

# # figura lana
# plt.figure(figsize=(12,12)) 
# plt.subplot(121)
# plt.imshow(cv2.cvtColor(img1_pb, cv2.COLOR_BGR2RGB), cmap="gray", vmin=0, vmax=255)
# plt.title("original image")
# plt.axis('off')
# plt.subplot(122)
# plt.imshow(cv2.cvtColor(img1_diff, cv2.COLOR_BGR2RGB), cmap="gray", vmin=0, vmax=255)
# plt.title("image filtered with differential filter")
# plt.axis('off')
# plt.show()

# figura goblin machine
# img2_diff = image_convolution(img2_pb, w_diff)

# plt.figure(figsize=(12,12)) 
# plt.subplot(121)
# plt.imshow(cv2.cvtColor(img2_pb, cv2.COLOR_BGR2RGB), cmap="gray", vmin=0, vmax=255)
# plt.title("original image")
# plt.axis('off')
# plt.subplot(122)
# plt.imshow(cv2.cvtColor(img2_diff, cv2.COLOR_BGR2RGB), cmap="gray", vmin=0, vmax=255)
# plt.title("image filtered with differential filter")
# plt.axis('off')
# plt.show()

# figura lana
# w_vert = np.matrix([[-1, 0, 1], 
#                     [-1, 0, 1], 
#                     [-1, 0, 1]])
# print(w_vert)

# img1_vert = image_convolution(img1_pb, w_vert)

# exibindo imagem 1 e filtrada por w_diff
# plt.figure(figsize=(12,12)) 
# plt.subplot(121)
# plt.imshow(img1_pb, cmap="gray", vmin=0, vmax=255)
# plt.title("imagem 1")
# plt.axis('off')
# plt.subplot(122)
# plt.imshow(img1_vert, cmap="gray", vmin=0, vmax=255)
# plt.title("imagem 1 convoluída com filtro diferencial vertical")
# plt.axis('off')
# plt.show()

# exibindo imagem 2 e filtrada por w_diff
# img2_vert = image_convolution(img2_pb, w_vert)

# plt.figure(figsize=(12,12)) 
# plt.subplot(121)
# plt.imshow(img2_pb, cmap="gray", vmin=0, vmax=255)
# plt.title("imagem 1")
# plt.axis('off')
# plt.subplot(122)
# plt.imshow(img2_vert, cmap="gray", vmin=0, vmax=255)
# plt.title("imagem 2 convoluída com filtro diferencial vertical")
# plt.axis('off')
# plt.show()

# blur = cv2.GaussianBlur(img1_pb,(3,3),0)

# #Filtro passa alta de Laplace.
# laplacian = cv2.Laplacian(img1_pb,cv2.CV_64F)

# plt.figure(figsize=(12,12)) 
# plt.subplot(131)
# plt.imshow(img1_pb, cmap="gray", vmin=0, vmax=255)
# plt.title("Original")
# plt.subplot(132)
# plt.imshow(blur, cmap="gray", vmin=0, vmax=255)
# plt.title("BLur")
# plt.axis('off')
# plt.subplot(133)
# plt.imshow(laplacian, cmap="gray", vmin=0, vmax=255)
# plt.title("Laplace")
# plt.axis('off')
# plt.show()

# blur = cv2.GaussianBlur(img2_pb,(3,3),0)

# #Filtro passa alta de Laplace.
# laplacian = cv2.Laplacian(img2_pb,cv2.CV_64F)

# plt.figure(figsize=(12,12)) 
# plt.subplot(131)
# plt.imshow(img2_pb, cmap="gray", vmin=0, vmax=255)
# plt.title("Original")
# plt.subplot(132)
# plt.imshow(blur, cmap="gray", vmin=0, vmax=255)
# plt.title("BLur")
# plt.axis('off')
# plt.subplot(133)
# plt.imshow(laplacian, cmap="gray", vmin=0, vmax=255)
# plt.title("Laplace")
# plt.axis('off')
# plt.show()

# img_sobelx = cv2.Sobel(img1_pb,cv2.CV_8U,1,0,ksize=5)
# img_sobely = cv2.Sobel(img1_pb,cv2.CV_8U,0,1,ksize=5)
# sobel = img_sobelx + img_sobely

# plt.figure(figsize=(12,12)) 
# plt.subplot(122)
# plt.imshow(sobel, cmap="gray", vmin=0, vmax=255)
# plt.title("Sobel")
# plt.axis('off')

# plt.subplot(121)
# plt.imshow(img1_pb, cmap="gray", vmin=0, vmax=255)
# plt.title("Original")
# plt.show()

# img_sobelx = cv2.Sobel(img2_pb,cv2.CV_8U,1,0,ksize=5)
# img_sobely = cv2.Sobel(img2_pb,cv2.CV_8U,0,1,ksize=5)
# sobel = img_sobelx + img_sobely

# plt.figure(figsize=(12,12)) 
# plt.subplot(122)
# plt.imshow(sobel, cmap="gray", vmin=0, vmax=255)
# plt.title("Sobel")
# plt.axis('off')

# plt.subplot(121)
# plt.imshow(img2_pb, cmap="gray", vmin=0, vmax=255)
# plt.title("Original")
# plt.show()

# kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
# kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
# #ao inves de usar a funcao convolucional dos exemplos acima, vc define o valor do filtro e usa no filter2D
# img_prewittx = cv2.filter2D(img1_pb, -1, kernelx)
# img_prewitty = cv2.filter2D(img1_pb, -1, kernely)
# img_prewitt = img_prewittx + img_prewitty

# plt.figure(figsize=(12,12)) 
# plt.subplot(121)
# plt.imshow(img1_pb, cmap="gray", vmin=0, vmax=255)
# plt.title("Original")
# plt.axis('off')
# plt.subplot(122)
# plt.imshow(img_prewitt, cmap="gray", vmin=0, vmax=255)
# plt.title("Prewitt")
# plt.show()

# kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
# kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
# #ao inves de usar a funcao convolucional dos exemplos acima, vc define o valor do filtro e usa no filter2D
# img_prewittx = cv2.filter2D(img2_pb, -1, kernelx)
# img_prewitty = cv2.filter2D(img2_pb, -1, kernely)
# img_prewitt = img_prewittx + img_prewitty

# plt.figure(figsize=(12,12)) 
# plt.subplot(121)
# plt.imshow(img2_pb, cmap="gray", vmin=0, vmax=255)
# plt.title("Original")
# plt.axis('off')
# plt.subplot(122)
# plt.imshow(img_prewitt, cmap="gray", vmin=0, vmax=255)
# plt.title("Prewitt")
# plt.show()

# edges = cv2.Canny(img1_pb,100,200)
# fig, ax = plt.subplots(ncols=2,figsize=(15,5))
# ax[0].imshow(img1_pb,cmap = 'gray')
# ax[0].set_title('Original Image') 
# ax[0].axis('off')
# ax[1].imshow(edges,cmap = 'gray')
# ax[1].set_title('Edge Image')
# ax[1].axis('off')
# plt.show()

# edges = cv2.Canny(img2_pb,100,200)
# fig, ax = plt.subplots(ncols=2,figsize=(15,5))
# ax[0].imshow(img2_pb,cmap = 'gray')
# ax[0].set_title('Original Image') 
# ax[0].axis('off')
# ax[1].imshow(edges,cmap = 'gray')
# ax[1].set_title('Edge Image')
# ax[1].axis('off')
# plt.show()