#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 13:52:52 2020

@author: joanna
"""
# import os
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from keras.optimizers import SGD


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# mnist = tf.keras.datasets.mnist
# (x_train, y_train),(x_test, y_test) = mnist.load_data()


# #normalizando as entradas
# x_train1 = tf.keras.utils.normalize(x_train, axis=1)
# x_test1  = tf.keras.utils.normalize(x_test, axis=1)

# print("x_train = ",x_train.shape)
# print("y_train = ",y_train.shape)
# plt.imshow(x_train[0], cmap=plt.cm.binary)
# plt.show()
# # print(x_train[0])

# #constroindo o modelo
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Dense(128, input_shape=(784,), activation="sigmoid"))
# model.add(tf.keras.layers.Dense(32, activation = tf.nn.relu))
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
# # model.add(tf.keras.layers.Dense(64, activation="sigmoid"))
# model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


# #model.compile(loss=keras.losses.categorical_crossentropy,
# #              optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True))

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# history= model.fit(x_train, y_train, epochs=3)

# val_loss, val_acc = model.evaluate(x_test, y_test)
# print(val_loss, val_acc)
# model.summary()

# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_acc'])
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.show()
# # plotar loss e accuracy para os datasets 'train' e 'test'
# # plt.style.use("ggplot")
# # plt.figure()
# # plt.plot(np.arange(0,100), H.history["loss"], label="train_loss")
# # plt.plot(np.arange(0,100), H.history["val_loss"], label="val_loss")


###################################################################
import keras
from keras.datasets import mnist
import tensorflow as tf
from keras.layers import Dense # Dense layers are "fully connected" layers
from keras.models import Sequential # Documentation: https://keras.io/models/sequential/

import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()# Setup train and test splits
print("Training data shape: ", x_train.shape) # (60000, 28, 28) -- 60000 images, each 28x28 pixels
print("Test data shape", x_test.shape) # (10000, 28, 28) -- 10000 images, each 28x28

# Flatten the images
image_vector_size = 28*28
x_train = x_train.reshape(x_train.shape[0], image_vector_size)
x_test = x_test.reshape(x_test.shape[0], image_vector_size)

# Convert to "one-hot" vectors using the to_categorical function
num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

print("First 5 training lables as one-hot encoded vectors:\n", y_train[:5])

image_size = 784 # 28*28
num_classes = 10 # ten unique digits

model = Sequential()

# The input layer requires the special input_shape parameter which should match
# the shape of our training data.
model.add(Dense(units=32, activation='sigmoid', input_shape=(image_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.summary()

model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=128, epochs=5, verbose=False, validation_split=.1)
loss, accuracy  = model.evaluate(x_test, y_test, verbose=False)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()

print(f'Test loss: {loss:.3}')
print(f'Test accuracy: {accuracy:.3}')










