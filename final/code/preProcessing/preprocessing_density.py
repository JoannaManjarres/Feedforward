#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 13:22:02 2020

@author: joanna
"""
import csv

import pandas as pd

#------------Convierte un txt em csv ------------
filename = '../../data/raw/input/coordinates/run00000/sumoOutputInfoFileName.txt' 
filename1 = '../../data/raw/input/coordinates/run00000/sumoOutputInfoFileName1.csv'
  
# g = open(filename1, "w")
# f = open(filename, "r")
# for x in f:
#     print(x) 
#     g.write(x)
# g.close()
# f.close()
#-----------------------------------------------------
car_length_inMeters = 4.645
trunk_length_inMeters = 12.5
bus_length_inMeters = 9.0
vehicle_high = 1.7

with open(filename1) as csvfile:
    reader = csv.DictReader(csvfile)
    id_episode_vector=[]
    type_car_vector=[]
    coord_x_vector=[]
    coord_y_vector=[]
    coordenadas =[]
    coordenadas1 =[]
    for row in reader:
        id_episode = row['episode_i']
        type_car = row['typeID']
        
        if type_car == 'Car':
            car_area = car_length_inMeters * vehicle_high
        elif type_car == 'Bus':
            bus_area = bus_length_inMeters
        elif type_car == 'Truck':
            trunk_area = trunk_length_inMeters
        
        
        
        coor_x=row['xinsite']
        coor_y=row['yinsite']
        
        id_episode_vector.append(id_episode)
        type_car_vector.append(type_car)
        coord_x_vector.append(coor_x)
        coord_y_vector.append(coor_y)

    coordenadas.append(id_episode_vector)
    coordenadas.append(type_car_vector)
    coordenadas.append(coord_x_vector)
    coordenadas.append(coord_y_vector)
    
    print(coord_x_vector)
        
