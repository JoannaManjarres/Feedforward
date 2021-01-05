

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import scipy.stats
import seaborn as sb


def obtener_coordenadas_validas():
    filename = "../../data/CoordVehiclesRxPerScene_s008.csv"
    limit_ep_train = 1564

    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        VehicleArrayID=[]
        VehicleName=[]
        rays=[]
        LOS=[]
        x_coordinates=[]
        y_coordinates=[]
        z_coordinates=[]

        x_coordinates_train = []
        y_coordinates_train = []
        z_coordinates_train = []

        LOS_int=[]


        for row in reader:
            is_valid = row['Val']  # canal valido -> V o Canal invalido-> I
            if is_valid == 'V':  # check if the channel is valid
                if int(row['EpisodeID']) <= limit_ep_train:
                    x_coordinates_train.append(float(row['x']))
                    y_coordinates_train.append(float(row['y']))
                    z_coordinates_train.append(float(row['z']))
                    VehicleArrayID.append(int(row['VehicleArrayID']))
                    VehicleName.append(row['VehicleName'])
                    rays.append(int(row['rays']))
                    LOS.append(row['LOS'])
                if int(row['EpisodeID']) > limit_ep_train:
                    x_coordinates.append(int(row['VehicleArrayID']))
                    y_coordinates.append(int(row['VehicleArrayID']))
                    z_coordinates.append(int(row['VehicleArrayID']))

    for i in LOS:
        if i =="LOS=0":
            LOS_int.append(0)
        else:
            LOS_int.append(1)

    return VehicleArrayID, VehicleName, rays, LOS, x_coordinates_train, y_coordinates_train, z_coordinates_train,LOS_int

def read_labels_data():
    label_path = "../../data/processed/output_beam/"

    input_cache_file = np.load(label_path + "beams_output_train.npz")
    label_train = input_cache_file["output_training"]

    input_cache_file = np.load(label_path + "beams_output_validation.npz")
    label_validation = input_cache_file["output_test"]

    return label_train, label_validation

def calc_coef_pearson(data):

    xticks_label = ["x", "y", "z", "Rays","LOS", "labels"]
    print(np.corrcoef(data))
    a = np.corrcoef(data)

    sb.heatmap(a,
               xticklabels=xticks_label,
               yticklabels=xticks_label,
               cmap='Blues',
               annot=True,
               linewidth=0.5)
    plt.show()


VehicleArrayID, VehicleName, rays, LOS, x_coordinates, y_coordinates, z_coordinates,LOS = obtener_coordenadas_validas()
label_train, label_validation = read_labels_data()
data = [x_coordinates, y_coordinates, z_coordinates, rays, LOS, label_train]
calc_coef_pearson(data)



