import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# exercicio 1
figura1_vetor = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=np.uint16)

figura1 = figura1_vetor * 255
estruturante_a = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
estruturante_b = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

# i.
erosao_a = cv2.erode(figura1, estruturante_a, iterations=1)
# ii.
erosao_b = cv2.erode(figura1, estruturante_b, iterations=1)
# iii.
dilatacao_a = cv2.dilate(figura1, estruturante_a, iterations=1)
# iv.
dilatacao_b = cv2.dilate(figura1, estruturante_b, iterations=1)

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].imshow(figura1, cmap='gray')
axs[0, 0].set_title('Original')
axs[0, 0].axis('off')

axs[0, 1].imshow(erosao_a, cmap='gray')
axs[0, 1].set_title('i. Erosão com SE 2(a)')
axs[0, 1].axis('off')

axs[0, 2].imshow(erosao_b, cmap='gray')
axs[0, 2].set_title('ii. Erosão com SE 2(b)')
axs[0, 2].axis('off')

axs[1, 0].imshow(figura1, cmap='gray')
axs[1, 0].set_title('Original')
axs[1, 0].axis('off')

axs[1, 1].imshow(dilatacao_a, cmap='gray')
axs[1, 1].set_title('iii. Dilatação com SE 2(a)')
axs[1, 1].axis('off')

axs[1, 2].imshow(dilatacao_b, cmap='gray')
axs[1, 2].set_title('iv. Dilatação com SE 2(b)')
axs[1, 2].axis('off')

plt.tight_layout()
plt.savefig("exercicio1_resultado.png")
plt.show()

# exercicio 2
squares = cv2.imread("quadrados.png", cv2.IMREAD_GRAYSCALE)
estruturante = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
img_erosao = cv2.erode(squares, estruturante, iterations=1)
img_dilatacao = cv2.dilate(img_erosao, estruturante, iterations=1)
fig, ax = plt.subplots(1, 3, figsize=(15, 10))

ax[0].imshow(squares, cmap='gray')
ax[0].set_title('Original')
ax[0].axis('off')

ax[1].imshow(img_erosao, cmap='gray')
ax[1].set_title('Erosão')
ax[1].axis('off')

ax[2].imshow(img_dilatacao, cmap='gray')
ax[2].set_title('Dilatação')
ax[2].axis('off')

plt.tight_layout()
plt.savefig("exercicio2_resultado.png")
plt.show()

# exercicio 3
ruidos = cv2.imread("ruidos.png", cv2.IMREAD_GRAYSCALE)
estruturante_abertura = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
estruturante_fechamento = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
img_abertura = cv2.morphologyEx(ruidos, cv2.MORPH_OPEN, estruturante_abertura)
img_fechamento = cv2.morphologyEx(ruidos, cv2.MORPH_CLOSE, estruturante_fechamento)
fig, ax = plt.subplots(1, 3, figsize=(15, 10))

ax[0].imshow(ruidos, cmap='gray')
ax[0].set_title('Original')
ax[0].axis('off')

ax[1].imshow(img_abertura, cmap='gray')
ax[1].set_title('Após Abertura (Ruído de Fundo Removido)')
ax[1].axis('off')

ax[2].imshow(img_fechamento, cmap='gray')
ax[2].set_title('Após Fechamento (Ruído do Objeto Removido)')
ax[2].axis('off')

plt.tight_layout()
plt.savefig("exercicio3_resultado.png")
plt.show()

# exercicio 4
dog = cv2.imread("cachorro.png", cv2.IMREAD_GRAYSCALE)
estruturante = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
img_dilatacao = cv2.dilate(dog, estruturante, iterations=1)
img_erosao = cv2.erode(dog, estruturante, iterations=1)
fronteira_externa = img_dilatacao - dog
fronteira_interna = dog - img_erosao

fig, ax = plt.subplots(1, 3, figsize=(15, 10))

ax[0].imshow(dog, cmap='gray')
ax[0].set_title('Original')
ax[0].axis('off')

ax[1].imshow(fronteira_externa, cmap='gray')
ax[1].set_title('Fronteira Externa')
ax[1].axis('off')

ax[2].imshow(fronteira_interna, cmap='gray')
ax[2].set_title('Fronteira Interna')
ax[2].axis('off')

plt.tight_layout()
plt.savefig("exercicio4_resultado.png")
plt.show()

# exercicio 5
cat = cv2.imread("gato.png", cv2.IMREAD_GRAYSCALE)

semente = np.zeros_like(cat)
semente[100, 100] = 255

img_complemento = cv2.bitwise_not(cat)
estruturante = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
img_anterior = np.zeros_like(cat)
img_atual = semente.copy()

while True:
    img_dilatada = cv2.dilate(img_atual, estruturante, iterations=1)
    img_proximo = cv2.bitwise_and(img_dilatada, img_complemento)
    if np.array_equal(img_proximo, img_atual):
        break
    img_atual = img_proximo

img_preenchida = cv2.bitwise_or(cat, img_atual)
fig, ax = plt.subplots(1, 2, figsize=(12, 8))

ax[0].imshow(cat, cmap='gray')
ax[0].set_title('Original')
ax[0].axis('off')

ax[1].imshow(img_preenchida, cmap='gray')
ax[1].set_title('Região Interna Preenchida')
ax[1].axis('off')

plt.tight_layout()
plt.savefig("exercicio5_resultado.png")
plt.show()

# exercicio 6
squares = cv2.imread("quadrados.png", cv2.IMREAD_GRAYSCALE)

ponto_inicial = (200, 150)
print(squares[ponto_inicial[1], ponto_inicial[0]])
h, w = squares.shape[:2]
mascara = np.zeros((h + 2, w + 2), np.uint8)

cv2.floodFill(squares, mascara, ponto_inicial, 255, 0, 0, cv2.FLOODFILL_MASK_ONLY)

regiao_isolada = mascara[1:h + 1, 1:w + 1]
img_resultado = np.zeros((h, w, 3), np.uint8)
mascara_3canais = cv2.merge([regiao_isolada, regiao_isolada, regiao_isolada])
img_resultado = np.where(mascara_3canais == 1, [0, 255, 255], img_resultado).astype(np.uint8)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.imshow(cv2.cvtColor(img_resultado, cv2.COLOR_BGR2RGB))
ax.set_title(f'Componente Conectada (Quadrado de 80px) em Amarelo')
ax.axis('off')

plt.tight_layout()
plt.savefig("exercicio6_resultado.png")
plt.show()

# exercicio 7
thur = cv2.imread("thur.jpg", cv2.IMREAD_GRAYSCALE)

estruturante = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

img_dilatacao = cv2.dilate(thur, estruturante, iterations=1)
img_erosao = cv2.erode(thur, estruturante, iterations=1)
img_gradiente = cv2.subtract(img_dilatacao, img_erosao)

fig, ax = plt.subplots(2, 2, figsize=(10, 10))

ax[0, 0].imshow(thur, cmap='gray')
ax[0, 0].set_title('Original')
ax[0, 0].axis('off')

ax[0, 1].imshow(img_dilatacao, cmap='gray')
ax[0, 1].set_title('Dilatação')
ax[0, 1].axis('off')

ax[1, 0].imshow(img_erosao, cmap='gray')
ax[1, 0].set_title('Erosão')
ax[1, 0].axis('off')

ax[1, 1].imshow(img_gradiente, cmap='gray')
ax[1, 1].set_title('Gradiente Morfológico')
ax[1, 1].axis('off')

plt.tight_layout()
plt.savefig("exercicio7_resultado.png")
plt.show()