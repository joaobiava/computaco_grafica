import cv2
from matplotlib import pyplot as plt
import numpy as np
import math

img = cv2.imread('imagens/lena.png')
img2 = cv2.imread('imagens/img500x500.jpg')

# cv2.imshow('sim', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('sim', img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cinzalena = img[:, :, 0]//3 + img[:, :, 1]//3 + img[:, :, 2]//3
# cv2.imshow('sim', cinzalena)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cinza500x500 = img2[:, :, 0]//3 + img2[:, :, 1]//3 + img2[:, :, 2]//3
# cv2.imshow('sim', cinza500x500)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# lenaNegativo = 255 - img
# cv2.imshow('sim', lenaNegativo)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# negativo500x500 = 255 - img2
# cv2.imshow('sim', negativo500x500)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# lena_normal = cv2.normalize(img, None, 0, 100, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# cv2.imshow('sim', lena_normal)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# normal_500x500 = cv2.normalize(img2, None, 0, 100, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# cv2.imshow('sim', normal_500x500)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img_float = img.astype(np.float32)
# c = 255/np.log(1 + np.max(img_float))
# loglena = c * np.log(1 + img_float)
# loglena = np.uint8(np.clip(loglena, 0, 255))

# cv2.imshow('sim', loglena)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img2_float = img2.astype(np.float32)
# c = 255/np.log(1 + np.max(img2_float))
# log500x500 = c * np.log(1 + img2_float)
# log500x500 = np.uint8(np.clip(log500x500, 0, 255))

# cv2.imshow('sim', log500x500)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img_normalized = img.astype(np.float32) / 255.0
# potenciaLena = 2 * (img_normalized**2)
# gamma_corrected = np.uint8(np.clip(potenciaLena * 255, 0, 255))
# cv2.imshow('sim', gamma_corrected)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img_normalized = img2.astype(np.float32) / 255.0
# potenciaLena = 2 * (img_normalized**2)
# gamma_corrected = np.uint8(np.clip(potenciaLena * 255, 0, 255))
# cv2.imshow('sim', gamma_corrected)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# def bit_planes(img, title):
#     plt.figure(figsize=(12,6))
#     for i in range(8):  # 8 bits
#         bit_plane = (img >> i) & 1  # extrai o bit i
#         plt.subplot(2,4,i+1)
#         plt.imshow(bit_plane*255, cmap='gray')
#         plt.title(f'Bit {i}')
#         plt.axis('off')
#     plt.suptitle(f'Planos de bits - {title}')
#     plt.show()

# bit_planes(img, "Lena")
# bit_planes(img2, "Img Aluno")

def histograma(img):
    h = np.zeros(256, dtype=int)
    for val in img.ravel():
        h[val] += 1
    return h

def histograma_normalizado(img):
    h = histograma(img)
    return h / h.sum()

def histograma_acumulado(img):
    h = histograma(img)
    return np.cumsum(h)

def histograma_acumulado_normalizado(img):
    h_norm = histograma_normalizado(img)
    return np.cumsum(h_norm)

# (i) Converte para cinza e gera histograma
unequalized = cv2.imread('imagens/img500x500.jpg', cv2.IMREAD_GRAYSCALE)
h_unequalized = histograma(unequalized)

plt.figure()
plt.plot(h_unequalized)
plt.title("Histograma - unequalized (cinza)")
plt.show()

# (ii) Histograma R, G e B
img_aluno_color = cv2.imread('imagens/img500x500.jpg')
colors = ('b','g','r')
plt.figure()
for i,col in enumerate(colors):
    h = histograma(img_aluno_color[:,:,i])
    plt.plot(h, color=col)
plt.title("Histogramas R, G e B - imagens/img500x500.jpg")
plt.show()

# (iii) Converte img_aluno para cinza e gera A, B, C, D
img_aluno_gray = cv2.cvtColor(img_aluno_color, cv2.COLOR_BGR2GRAY)

hA = histograma(img_aluno_gray)
hB = histograma_normalizado(img_aluno_gray)
hC = histograma_acumulado(img_aluno_gray)
hD = histograma_acumulado_normalizado(img_aluno_gray)

plt.figure(figsize=(10,8))
plt.subplot(2,2,1); plt.plot(hA); plt.title("A. Histograma")
plt.subplot(2,2,2); plt.plot(hB); plt.title("B. Normalizado")
plt.subplot(2,2,3); plt.plot(hC); plt.title("C. Acumulado")
plt.subplot(2,2,4); plt.plot(hD); plt.title("D. Acumulado Normalizado")
plt.show()

def equalizacao_histograma(img):
    h = histograma(img)
    h_ac = np.cumsum(h)
    # normalizar para 0–255
    h_eq = np.round((h_ac - h_ac.min()) * 255 / (h_ac.max() - h_ac.min())).astype('uint8')
    img_eq = h_eq[img]
    return img_eq

# Aplicar
for nome in ['imagens/lena.png', 'imagens/img500x500.jpg']:
    img = cv2.imread(nome, cv2.IMREAD_GRAYSCALE)
    img_eq = equalizacao_histograma(img)

    cv2.imshow(f"Original - {nome}", img)
    cv2.imshow(f"Equalizada - {nome}", img_eq)
    cv2.waitKey(0)

cv2.destroyAllWindows()
