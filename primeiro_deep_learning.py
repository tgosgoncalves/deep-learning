import numpy  #importa biblioteca para importação de matrizes
from matplotlib import pyplot as plt #importa biblioteca para criação de gráficos

# início das importações de classes do Kersas (modelo de deep learning)

from keras.datasets import mnist  # importa o dataset Mnist (modelo que quero usar)
from keras.models import Sequential  # impotta modelo sequencial
from keras.layers import Dense  # importa camadas totalmente conectadas
from keras.layers import Dropout  # importa a estrutura de dropout (camadas)
from keras.utils import np_utils  # importa biblioteca de utilidades do Keras (vetor, matriz, etc)

(x_train, y_train), (x_test, y_test) = mnist.load_data()  #i importa o dataset Mnist da biblioteca do Keras
print(x_train.shape)  # imprime o tamanho do vetor

first_image = x_train[100]  # obtém a imagem de treino no indice 100 do vetor

#realiza a manipulação dos dados da imagem
first_image = numpy.array(first_image, dtype='float')
pixels = first_image.reshape((28, 28))
#exibe a imagem do vetor
plt.imshow(pixels, cmap='gray')
plt.show()

print(y_train[100])  #imprime a classe da imagem de treino do indice 100