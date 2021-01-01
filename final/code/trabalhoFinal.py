#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 14:42:03 2020

@author: joanna
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.optimizers import SGD

def read_labels_data(debug=False):
    if(debug):
         labelPath = "../data/processed/output_beam/debug/"
    else:
        labelPath = "../data/processed/output_beam/"
    
    input_cache_file = np.load(labelPath+"beams_output_train.npz")
    label_train = input_cache_file["output_training"]
                                          
    input_cache_file = np.load(labelPath+"beams_output_validation.npz")
    label_validation = input_cache_file["output_test"]
    
    print("Label train: ",label_train)
    print("Label train: ",type(label_train))
    print("Label train: ",label_train.shape)
    print(label_train.shape)
    
    return label_train, label_validation

def read_inputs(multiple_inputs=False, debug=False):
    if(debug):
        coord_input_path = "../data/processed/coord_input/debug/"
        images_input_path = "../data/processed/images_input/debug/"
    else:
        coord_input_path = "../data/processed/coord_input/"
        images_input_path = "../data/processed/images_input/"
        
    input_cache_file = np.load(coord_input_path+"coord_train.npz")
    input_train      = input_cache_file["coordinates_train"]
    
    input_cache_file = np.load(coord_input_path+"coord_test.npz")
    input_validation = input_cache_file["coordinates_validation"]
    
    print("input Train: ",input_train.shape)
    # print(input_train[0].shape)
    return input_train, input_validation
    
def simple_network(input_train,label_train, input_validation, label_validation):
    #constroindo o modelo
    nodes_input_layer = input_train.shape[0]
    nodes_output_layer = label_train.shape[0]
    model = tf.keras.models.Sequential()
    # model.add(tf.keras.layers.Flatten())
    #capa densamente conectada, "completamente conectada"
    model.add(tf.keras.layers.Dense(128, activation = tf.nn.sigmoid))
    
    model.add(tf.keras.layers.Dense(30, activation = tf.nn.relu))
    model.add(tf.keras.layers.Dense(nodes_output_layer, activation=tf.nn.softmax))
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
              # loss='sparse_categorical_crossentropy',
              
    history = model.fit(input_train, label_train, epochs=5)
    val_loss, val_acc = model.evaluate(input_validation, label_validation)
    print(val_loss, val_acc)
    
    plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_acc'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()

input_train,input_validation = read_inputs(debug=False)
label_train,label_validation = read_labels_data(debug=False)
simple_network(input_train,label_train, input_validation, label_validation)  
    
