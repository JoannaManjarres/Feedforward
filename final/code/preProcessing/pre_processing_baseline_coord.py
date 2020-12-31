#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 12:17:39 2020

@author: joanna
"""
import numpy as np
import csv

test=0

def load_baseline_data():
    
    baseline_path = "../../data/processed/coord_input/baseline/"

    print('\n load coord baseline training data ...')
    coord_train_input_file = baseline_path+'coord_train.npz'
    coord_train_cache_file = np.load(coord_train_input_file)
    X_coord_train = coord_train_cache_file['coordinates']
    coord_train_input_shape = X_coord_train.shape
    
    print('\n load coord baseline validation data ...')
    coord_validation_input_file = baseline_path+'coord_validation.npz'
    coord_validation_cache_file = np.load(coord_validation_input_file)
    X_coord_validation = coord_validation_cache_file['coordinates']
    coord_validation_input_shape = X_coord_validation.shape

    print('\n coord training shape: ', coord_train_input_shape)
    print('\n coord validation shape: ', coord_validation_input_shape)
    return X_coord_train, X_coord_validation


def obtener_coordenadas_validas():

    filename = "../../data/processed/coord_input/CoordVehiclesRxPerScene_s008.csv"
    limit_ep_train = 1564
    
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        coordinates_train = []
        coordinates_test = []      
        id_episodio_train =[]
        id_episodio_test =[]
        
        for row in reader:
            isValid = row['Val'] #canal valido -> V o Canal invalido-> I 
            if isValid == 'V': #check if the channel is valid
                if int(row['EpisodeID']) <= limit_ep_train:
                    id_episodio_train.append(int(row['EpisodeID']))
                    coordinates_train.append([int(row['EpisodeID']),float(row['x']),float(row['y']),float(row['z'])])
                if int(row['EpisodeID']) > limit_ep_train:
                    coordinates_test.append([int(row['EpisodeID']),float(row['x']),float(row['y']),float(row['z'])])
                    id_episodio_test.append(int(row['EpisodeID']))
    
    return coordinates_train, coordinates_test, limit_ep_train, id_episodio_train, id_episodio_test


def preprocesamiento_coordenadas(coordenadas, limit_ep_train, tipo_coor):
    
    offset_inCoord=[658.82,358.76] 
    interestAreaSize_inCoord=[181.26,317.14] 
    interestAreaSize_inMeters = [23, 250]   
    carrLength_inMeters = 4.645
    trunkLength_inMeters = 12.5
    busLength_inMeters = 9.0
    
    coordTometers_y = 250/317.14
    metersToCoord_y = 1/coordTometers_y
    coordTometers_x = 23/181.26
    metersToCoord_x = 1/coordTometers_x
    
    yExtension_inCoord = interestAreaSize_inCoord[0]
    
    max_length_inCoord = interestAreaSize_inCoord[0]
    max_hight_inCoord = interestAreaSize_inCoord[1]+yExtension_inCoord
    offset_x_inCoord = offset_inCoord[0]
    offset_y_inCoord = offset_inCoord[1]

    # coord_train1, coord_validation1, limit_ep_train, _, _ = obtener_coordenadas_validas()

    coord_train = np.asarray(coordenadas, dtype=np.float32)
    # coord_validation = np.asarray(coord_validation1, dtype=np.float32)

    coord_train_x_inMeters = (coord_train[:,1]-offset_x_inCoord)/max_length_inCoord*interestAreaSize_inMeters[0]
    coord_train_y_inMeters = (coord_train[:,2]-offset_y_inCoord)/max_hight_inCoord*interestAreaSize_inMeters[1]
    coord_train_x_inMeters = np.round(coord_train_x_inMeters)
    coord_train_y_inMeters = np.round(coord_train_y_inMeters)

    numOfSamples = len(coord_train_x_inMeters)
    Qs_array = []
    
    tipo_coordenadas = tipo_coor
    ep_final=int(max(coord_train[:,0]))
    
    if(tipo_coordenadas == "train"):
        range_episode = range(limit_ep_train+1) #limit_ep_train=1564
    elif(tipo_coordenadas == "validation"):
        range_episode = range(limit_ep_train+1,ep_final+1)

    for cont_ep in range_episode: 
    # for cont_ep in range(2): 
        Qs = np.zeros(interestAreaSize_inMeters)
        numOfValidCommunications=0
        for cont_samples in range(numOfSamples):
            if cont_ep==int(coord_train[cont_samples,0]):
                numOfValidCommunications=numOfValidCommunications+1
                Qs[int(coord_train_x_inMeters[cont_samples]),
                    int(coord_train_y_inMeters[cont_samples])]=1
        for i in range(numOfValidCommunications):
            Qs_array.append(Qs)
    
    Qs_vector = np.asarray(Qs_array)
    
    num_positions_of_internal_matrix = Qs_vector.shape[1] * Qs_vector.shape[2]
    X_train = Qs_vector.reshape(Qs_vector.shape[0],num_positions_of_internal_matrix)
    

       
    # print ('Coord npz files saved!')
    
    return X_train

# def reorganizar dados para a rede

############################### Main ##################################
coord_train1, coord_validation1, limit_ep_train, _, _ = obtener_coordenadas_validas()

X_train = preprocesamiento_coordenadas(coord_train1, limit_ep_train,"train")
X_test = preprocesamiento_coordenadas(coord_validation1, limit_ep_train,"validation")    

X_train_debug = X_train[0:20,:]
X_test_debug = X_test[0:2,:]

saveInputPath = "../../data/processed/coord_input/"
debugPath = "../../data/processed/coord_input/debug/"
   
# #train
np.savez(saveInputPath+'coord_train'+'.npz',coordinates_train=X_train)
np.savez(debugPath+'coord_train'+'.npz',coordinates_train=X_train_debug)
# test
np.savez(saveInputPath+'coord_test'+'.npz',coordinates_validation=X_test)
np.savez(debugPath+'coord_test'+'.npz',coordinates_validation=X_test_debug)