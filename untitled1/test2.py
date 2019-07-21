from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import Flatten
from keras.layers import MaxPool2D
from keras.layers import Dropout
from keras.layers import Activation
from keras.utils import np_utils
from keras.layers import Deconv2D

model = Sequential()
(data, label), (x_test, y_test) = mnist.load_data()

print(data.shape)

x_train = data.reshape(data.shape[0], 28, 28, 1)
y_train = np_utils.to_categorical(label, 10)

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
y_test = np_utils.to_categorical(y_test, 10)

model.add(Convolution2D(25, 3, strides=(1, 1), input_shape=(28, 28, 1)))
model.add(MaxPool2D((2, 2)))

model.add(Convolution2D(50, 3, 3))
model.add(MaxPool2D((2, 2)))

model.add(Convolution2D(50, 3, 3))
model.add(MaxPool2D((2, 2)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(output_dim=100))
model.add(Activation('sigmoid'))

model.add(Dense(output_dim=10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=1000, nb_epoch=10)

res = model.evaluate(x_test, y_test)

print('total loss: ', res[0])
print('Accuracy: ', res[1])
