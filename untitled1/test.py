from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense
from keras.layers import Activation
from keras.utils import np_utils


model = Sequential()
(data, label), (x_test, y_test) = mnist.load_data()


x_train = data.reshape(len(data), -1)
y_train = np_utils.to_categorical(label, 10)

x_test = x_test.reshape(len(x_test), -1)

y_test = np_utils.to_categorical(y_test, 10)
model.add(Dense(input_dim=28*28, output_dim=500))
model.add(Activation('sigmoid'))

# model.add(Dense(output_dim=500))
# model.add(Activation('relu'))
#
# model.add(Dense(output_dim=500))
# model.add(Activation('relu'))

# model.add(Dense(output_dim=500))
# model.add(Activation('relu'))

model.add(Dense(output_dim=10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=100, nb_epoch=50)

res = model.evaluate(x_test, y_test)

print('total loss: ', res[0])
print('Accuracy: ', res[1])
