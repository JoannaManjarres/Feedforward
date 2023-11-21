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
import csv
from keras.optimizers import SGD

def read_labels_data(debug=False):
    if(debug):
         labelPath = "../data/processed/output_beam/debug/"
         print(labelPath)
    else:
        labelPath = "../data/processed/output_beam/baseline/"
        print(labelPath)
    
    input_cache_file = np.load(labelPath+"beams_output_train.npz")
 
    label_train = input_cache_file["output_training"]
    
                                          
    input_cache_file = np.load(labelPath+"beams_output_validation.npz")
    label_validation = input_cache_file["output_test"]
    
    print("Label train: ",label_train)
    print("Label train: ",type(label_train))
    print("Label train: ",label_train.shape)
    print(label_train.shape)
    
    return label_train, label_validation

def read_all_beams():
        # path = '/Users/Joanna/git/beam_selection_wisard/data/beams/'+antenna_config+'/all_index_beam/'

        path = '../data/processed/beams/'


        input_cache_file = np.load(path + "index_beams_combined_train.npz", allow_pickle=True)
        #index_beam_combined_train = input_cache_file["all_beam_combined_train"].astype(str)
        index_beam_combined_train = input_cache_file["all_beam_combined_train"]

        input_cache_file = np.load(path + "index_beams_combined_test.npz", allow_pickle=True)
        #index_beam_combined_test = input_cache_file["all_beam_combined_test"].astype(str)
        index_beam_combined_test = input_cache_file["all_beam_combined_test"]

        return index_beam_combined_train, index_beam_combined_test
def read_coord():
    #filename = "../data/coordinates/CoordVehiclesRxPerScene_s008.csv"
    filename ="../data/processed/coord_input/CoordVehiclesRxPerScene_s008.csv"
    limit_ep_train = 1564

    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        number_of_rows = len(list(reader))

    all_info_coord_val = np.zeros([11194, 5], dtype=object)

    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        cont = 0
        for row in reader:
            if row['Val'] == 'V':
                all_info_coord_val[cont] = int(row['EpisodeID']), float(row['x']), float(row['y']), float(row['z']), \
                row['LOS']
                cont += 1

    # all_info_coord = np.array(all_info_coord)

    # Separacao do conjunto de dados em treinamento e teste
    coord_train_all_info = all_info_coord_val[(all_info_coord_val[:, 0] < limit_ep_train + 1)]
    coord_test_all_info = all_info_coord_val[(all_info_coord_val[:, 0] > limit_ep_train)]

    coord_train = coord_train_all_info[:, 1]
    coord_test = coord_test_all_info[:, 1]

    coord_train = tf.keras.utils.normalize(coord_train)
    coord_test = tf.keras.utils.normalize(coord_test)




    return coord_train, coord_test
def read_inputs(multiple_inputs=False, debug=False):
    if(debug):
        coord_input_path = "../data/processed/coord_input/debug/"
        images_input_path = "../data/processed/images_input/debug/"
    else:
        coord_input_path = "../data/processed/coord_input/baseline/"
        images_input_path = "../data/processed/images_input/"
        
    input_cache_file = np.load(coord_input_path+"coord_train.npz")
    input_train      = input_cache_file["coordinates"]
    
    input_cache_file = np.load(coord_input_path+"coord_validation.npz")
    input_validation = input_cache_file["coordinates"]
    
    #print("input Train: ",input_train.shape)
    # print(input_train[0].shape)
    return input_train, input_validation
    
def simple_network(input_train, label_train, input_validation, label_validation):
    print("input train: ",input_train.shape)
    print("label train: ",label_train.shape)

    input_train = tf.convert_to_tensor(input_train, dtype = tf.float32)
    input_validation = tf.convert_to_tensor(input_validation, dtype = tf.float32)
    #input_validation = input_validation.tolist()

    label_train = np.expand_dims(label_train, axis=-1)
    #label_validation = tf.convert_to_tensor(label_validation, dtype = tf.float32)

    #constroindo o modelo
    #nodes_input_layer = input_train.shape[0]
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
              
    history = model.fit(input_train, label_train, epochs=5, validation_split=.1)
    # val_loss, val_acc = model.evaluate(input_validation, label_validation, verbose=False)
    model.evaluate(input_validation, label_validation, verbose=False)
    # loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
    # print(val_loss, val_acc)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='best')
    plt.show()


tf.keras.backend.clear_session()



input_train,input_validation = read_inputs(debug=False)
label_train,label_validation = read_all_beams()
#label_train,label_validation = read_labels_data(debug=False)
simple_network(input_train,label_train, input_validation, label_validation)  

# codigo nuevo
input_train,input_validation = read_coord()
label_train,label_validation = read_all_beams()
simple_network(input_train,label_train, input_validation, label_validation)
