# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 09:32:23 2021

@author: ELETRON
"""


import numpy as np
import pandas as pd
import csv
from keras.optimizers import SGD
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import normalize
import matplotlib.cm as cm

def obtener_coordenadas_validas():

    global x_lim_min
    global x_lim_max 
    global y_lim_min
    global y_lim_max 
    
    # filename = "../../data/processed/coord_input/CoordVehiclesRxPerScene_s008.csv"
    filename = "data/CoordVehiclesRxPerScene_s008.csv"
    limit_ep_train = 1564
    
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        coordinates_train = []
        coordinates_test = []      
        x_coordinates_train=[]
        y_coordinates_train=[]
        x_coordinates_test=[]
        y_coordinates_test=[]
        z_coordinates_train=[]
        z_coordinates_test=[]
        
        
        for row in reader:
            isValid = row['Val'] #canal valido -> V o Canal invalido-> I 
            if isValid == 'V': #check if the channel is valid
                if int(row['EpisodeID']) <= limit_ep_train:
                    # coordinates_train.append([int(row['EpisodeID']),float(row['x']),float(row['y']),float(row['z'])])
                    x_coordinates_train.append(float(row['x']))
                    y_coordinates_train.append(float(row['y']))
                    z_coordinates_train.append(float(row['z']))
                if int(row['EpisodeID']) > limit_ep_train:
                    coordinates_test.append([int(row['EpisodeID']),float(row['x']),float(row['y']),float(row['z'])])
                    x_coordinates_test.append(float(row['x']))
                    y_coordinates_test.append(float(row['y']))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x_coordinates_train, y_coordinates_train, alpha=0.25)
    plt.xlim([x_lim_min, x_lim_max]) 
    plt.ylim([y_lim_min, y_lim_max]) 
    
    #adds a title and axes labels
    ax.set_title('Coordenadas de Treinamento')
    ax.set_xlabel('coordenadas Y')
    ax.set_ylabel('coordenadas Y')
            
     #removing top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #adds major gridlines
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.scatter(x_coordinates_test, y_coordinates_test, alpha=0.3, color='grey')
    plt.xlim([x_lim_min, x_lim_max]) 
    plt.ylim([y_lim_min, y_lim_max])  
    
    #adds a title and axes labels
    ax1.set_title('Coordenadas de Test')
    ax1.set_xlabel('coordenadas Y')
    ax1.set_ylabel('coordenadas Y')
            
     #removing top and right borders
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    #adds major gridlines
    ax1.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    plt.show()
    
    
    
    return coordinates_train, coordinates_test, limit_ep_train




def plot_all_coordinates():
    
    global x_lim_min
    global x_lim_max 
    global y_lim_min
    global y_lim_max
    
        #loading dataset
    filename = "data/CoordVehiclesRxPerScene_s008.csv"
    df = pd.read_csv(filename)
    df.columns = ['Val','EpisodeID', 'SceneID', 'VehicleArrayID', 'VehicleName', 'x','y','z','rays','LOS']
            
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df['x'], df['y'])
    
    ax.set_title('Todas as Coordenadas')
    ax.set_xlabel('coordenadas Y')
    ax.set_ylabel('coordenadas Y')
    plt.xlim([x_lim_min, x_lim_max]) 
    plt.ylim([y_lim_min, y_lim_max]) 
    
    plt.show()
    
    return df


x_lim_min = 650
x_lim_max = 850
y_lim_min = 350
y_lim_max = 700

obtener_coordenadas_validas()
plot_all_coordinates()
