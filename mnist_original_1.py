#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 13:52:52 2020

@author: joanna
"""
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.optimizers import SGD


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()


#normalizando as entradas
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test  = tf.keras.utils.normalize(x_test, axis=1)

print("x_train = ",x_train.shape)
print("y_train = ",y_train.shape)
plt.imshow(x_train[0], cmap=plt.cm.binary)
plt.show()
# print(x_train[0])

#constroindo o modelo
model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Dense(128, input_shape=(784,), activation="sigmoid"))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
# model.add(tf.keras.layers.Dense(64, activation="sigmoid"))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# model.compile(optimizer ='adam',
#               loss ='categorical_crossentropy',
#               metrics = ['accuracy'])

## codigo que funciona
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)

# model.compile(optimizer=SGD(0.01),
#               loss = 'categorical_crossentropy',
#               metrics = ['accuracy'])
# model.fit(x_train, y_train, epochs=3)
# model.fit(x_train, y_train, batch_size=128, epochs=3, verbose=2,
#          validation_data=(x_test, y_test))

