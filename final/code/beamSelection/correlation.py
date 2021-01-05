

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
from scipy.stats import pearsonr
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

    correlation_matrix = sb.heatmap(a,
                                    xticklabels=xticks_label,
                                    yticklabels=xticks_label,
                                    cmap='Blues',
                                    annot=True,
                                    linewidth=0.5)
    # plt.show()
    figure = correlation_matrix.get_figure()
    figure.savefig('../../results/preliminary statistics/Matrix_corelation.png')
    plt.close()

    return a

def plot_correlation(data,coef_correlacion):

    coef_pearson_x = round(coef_correlacion[0,5], 3)
    coef_pearson_y = round(coef_correlacion[1,5], 3)
    coef_pearson_z = round(coef_correlacion[2,5], 3)
    coef_pearson_rays = round(coef_correlacion[3, 5], 3)
    coef_pearson_LOS = round(coef_correlacion[4, 5], 3)

    plt.scatter(data[0], data[-1], label="Coef Pearson: "+str(coef_pearson_x), alpha=0.3)
    plt.xlabel("Coordenada x")
    plt.ylabel("Saída")
    plt.savefig(path + '/correlation/coorX_label.png')
    plt.close()

    plt.scatter(data[1], data[-1], label="Coef Pearson: " + str(coef_pearson_y), alpha=0.3)
    plt.xlabel("Coordenada y")
    plt.ylabel("Saída")
    plt.savefig(path + '/correlation/coorY_label.png')
    plt.close()

    plt.scatter(data[2], data[-1], label="Coef Pearson: " + str(coef_pearson_z), alpha=0.3)
    plt.xlabel("Coordenada y")
    plt.ylabel("Saída")
    plt.savefig(path + '/correlation/coorZ_label.png')
    plt.close()

    plt.scatter(data[3], data[-1], label="Coef Pearson: " + str(coef_pearson_rays), alpha=0.3)
    plt.xlabel("número de raios")
    plt.ylabel("Saída")
    plt.savefig(path + '/correlation/ray.png')
    plt.close()

    plt.scatter(data[4], data[-1], label="Coef Pearson: " + str(coef_pearson_LOS), alpha=0.3)
    plt.xlabel("Linha de visado")
    plt.ylabel("Saída")
    plt.savefig(path + '/correlation/LOS.png')
    plt.close()


def plot_histogramas(data, path):

    plt.style.use('ggplot')
    plt.hist(data[-1], bins=20, rwidth=8, color='steelblue', alpha=0.5)
    plt.title("Labels")
    plt.savefig(path+'histograma/Hist_labels.png')
    plt.close()
    # plt.show()

    plt.style.use('ggplot')
    plt.hist(data[0], bins='scott', color='steelblue', alpha=0.5)
    plt.title("Coordenadas x")
    plt.savefig(path + 'histograma/Hist_x_coord.png')
    plt.close()
    # plt.show()

    plt.style.use('ggplot')
    plt.hist(data[1], bins='scott', color='steelblue', alpha=0.5)
    plt.title("Coordenadas y")
    plt.savefig(path + 'histograma/Hist_y_coord.png')
    plt.close()
    # plt.show()

    plt.style.use('ggplot')
    plt.hist(data[2], bins='scott', color='steelblue', alpha=0.5)
    plt.title("Coordenadas z")
    plt.savefig(path + 'histograma/Hist_z_coord.png')
    plt.close()
    # plt.show()

    plt.style.use('ggplot')
    plt.hist(data[3], bins='scott', color='steelblue', alpha=0.5)
    plt.title("Quantidade de Raios")
    plt.savefig(path + 'histograma/Hist_ray.png')
    plt.close()
    # plt.show()

    plt.style.use('ggplot')
    plt.hist(data[4], bins='scott', color='steelblue', alpha=0.5)
    plt.title("LOS")
    plt.savefig(path + 'histograma/Hist_LOS.png')
    plt.close()
    # plt.show()


path = '../../results/preliminary statistics/'
VehicleArrayID, VehicleName, rays, LOS, x_coordinates, y_coordinates, z_coordinates,LOS = obtener_coordenadas_validas()
label_train, label_validation = read_labels_data()
data = [x_coordinates, y_coordinates, z_coordinates, rays, LOS, label_train]
coef_corre = calc_coef_pearson(data)
plot_correlation(data,coef_corre)
plot_histogramas(data, path)

print(coef_corre.shape)

