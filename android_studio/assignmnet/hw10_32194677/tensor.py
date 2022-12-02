import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
import matplotlib.pyplot as plt


model = Sequential()

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 28*28).astype('float32')/255.0
X_test = X_test.reshape(10000, 28*28).astype('float32')/255.0
Y_train = keras.utils.to_categorical(Y_train, 10)
Y_test = keras.utils.to_categorical(Y_test, 10)
model.add(Dense(units=20, input_dim=(28*28), activation='sigmoid'))
model.add(Dense(units=24 , activation='sigmoid'))
model.add(Dense(units=10, activation='softmax'))
model.summary()

model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(X_train, Y_train, batch_size=10, epochs=60, validation_split=0.3)
loss, accuracy = model.evaluate(X_test, Y_test)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()