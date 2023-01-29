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

num_pixels = x_train.shape[1] * x_train.shape[2]  #Calcula o total de pixel da imagem
print(num_pixels)

# transforma os valores dos pixels para float32
x_train2 = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')
x_test2 = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')

#print(first_image)  printa a matriz do numero

# transfomando os valores do pixel entre 0 e 1 
x_train2 = x_train2 / 255
x_test2 = x_test2 / 255

#print(x_train[100])
#print(x_test[100])

#transforma os Y em one-hot vector
y_train_h = np_utils.to_categorical(y_train)
y_test_h = np_utils.to_categorical(y_test)

#obtém o numero de classes do problema
num_classes = y_test_h.shape[1]

print(num_pixels)
print(y_train[101])
print(y_train_h[101])

# cria um modelo do tipo sequncial
model = Sequential()
model.add(inputLayer(input_shape=num_pixels))  #cria a camada de entrada
model.add(Dense(1024, kernel_initializer='normal', activation='relu'))  #cria a primeira camada da rede
model.add(Dense(2048, kernel_initializer='normal', activation='relu'))  #cria a segunda camada da rede
model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))  #cria a segunda camada de saida (softmax transforma os valores de saida de um vetor de valores em uma distribuição de probabilidade

model.summary()  #imprime informações sobre o modelo

from numpy.core.multiarray import result_type
# prepara o modelo para ser executado
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  #computa a perda entre os labels e as predições o que deveria ter calculado vs o que foi calculado
result = model.fit(x_train2, y_train_h, validation_data=(x_test2, y_test_h), epochs = 20, verbose = 1, batch_size = 100)  #executa o treinamento

#obtem a imagem do numero 1001
x = x_train2[101]
print(x.shape)
x = numpy.expand_dims(x, axis=0)
print(x.shape)

#imprime a avaliação da amostra
print(model.predict(x))
print(numpy.argmax(model.predict(x)))

