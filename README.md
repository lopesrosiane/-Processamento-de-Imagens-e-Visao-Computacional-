# -Processamento-de-Imagens-e-Visao-Computacional-
from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    from keras.models import load_model # O TensorFlow é necessário para o funcionamento do Keras
import cv2 # Instale opencv-python
import numpy as np
# Desative a notação científica para maior clareza
np.set_printoptions(suppress=True)
# Carregue o modelo
modelo = load_model("keras_model.h5", compile=False)
# Carregue as etiquetas (rótulos)
nomes_classes = open("labels.txt", "r").readlines()
# A CÂMERA pode ser 0 ou 1, com base na câmera padrão do seu computador
camera = cv2.VideoCapture(1)
while True:
# Capture a imagem da webcam.
ret, imagem = camera.read()
# Redimensione a imagem bruta para (224-altura,224-largura) pixels
imagem = cv2.resize(imagem, (224, 224), interpolation=cv2.INTER_AREA)
# Exiba a imagem em uma janela
# cv2.imshow("Imagem da Webcam", imagem)
redimensionada = cv2.resize(imagem, (600, 400))
cv2.imshow("Tela da Turma UFOPA", redimensionada) # exiba o quadro/imagem
# Transforme a imagem em um array numpy e redimensione-a para o formato de entrada do modelo.
imagem = np.asarray(imagem, dtype=np.float32).reshape(1, 224, 224, 3)
# Normalize o array de imagem
imagem = (imagem / 127.5) - 1
# Faça a previsão com o modelo
previsao = modelo.predict(imagem)
indice = np.argmax(previsao)
nome_classe = nomes_classes[indice]
pontuacao_confianca = previsao[0][indice]
# Imprima a previsão e a pontuação de confiança
print("Classe:", nome_classe[2:], end="")
print("Pontuação de Confiança:", str(np.round(pontuacao_confianca * 100))[:-2], "%")
# Ouça o teclado para pressionamentos.
entrada_teclado = cv2.waitKey(1)
# 27 é o código ASCII para a tecla "esc" no seu teclado.
if entrada_teclado == 27:
        break

camera.release()
cv2.destroyAllWindows()
