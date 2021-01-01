#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 13:52:52 2020

@author: joanna
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.optimizers import SGD


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()


#normalizando as entradas
x_train1 = tf.keras.utils.normalize(x_train, axis=1)
x_test1  = tf.keras.utils.normalize(x_test, axis=1)

print("x_train = ",x_train.shape)
print("y_train = ",y_train.shape)
plt.imshow(x_train[0], cmap=plt.cm.binary)
plt.show()
# print(x_train[0])

#constroindo o modelo
model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Dense(128, input_shape=(784,), activation="sigmoid"))
model.add(tf.keras.layers.Dense(32, activation = tf.nn.relu))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
# model.add(tf.keras.layers.Dense(64, activation="sigmoid"))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


#model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history= model.fit(x_train, y_train, epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)
model.summary()

plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_acc'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()
# plotar loss e accuracy para os datasets 'train' e 'test'
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(0,100), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0,100), H.history["val_loss"], label="val_loss")
