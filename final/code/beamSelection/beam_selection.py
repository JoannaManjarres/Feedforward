#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 10:45:28 2020

@author: joanna
"""
import timeit
import sys
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import confusion_matrix_pretty_print as pretty

sys.path.append('../preProcessing/')
from pre_processing_baseline_output import process_and_save_output_beams


def read_labels_data(debug=False):
    if debug:
        label_path = "../../data/processed/output_beam/debug/"
    else:
        label_path = "../../data/processed/output_beam/"

    input_cache_file = np.load(label_path + "beams_output_train.npz")
    label_train = input_cache_file["output_training"]

    input_cache_file = np.load(label_path + "beams_output_validation.npz")
    label_validation = input_cache_file["output_test"]

    return label_train, label_validation


def read_inputs(debug=False):
    if debug:
        input_path = "../../data/processed/coord_input/debug/"
    else:
        input_path = "../../data/processed/coord_input/"

    input_cache_file = np.load(input_path + "coord_train.npz")
    train_data = input_cache_file["coordinates_train"]

    input_cache_file = np.load(input_path + "coord_test.npz")
    validation_data = input_cache_file["coordinates_validation"]

    return train_data, validation_data


# noinspection PyGlobalUndefined
def tic():
    global tic_s
    tic_s = timeit.default_timer()


# noinspection PyGlobalUndefined
def toc():
    global tic_s
    toc_s = timeit.default_timer()

    return toc_s - tic_s


def neural_network(x_train,
                   y_train,
                   x_validation):
    global model
    global epocas

    print('\n Training NN ...')
    tic()
    # train using the input data
    model.fit(x_train, y_train, epochs=epocas)

    tiempo_entrenamiento_ms = toc()

    print('\n Selecting Beams using NN ...')
    tic()
    # classify some data
    y_validation = np.argmax(model.predict(x_validation), axis=-1)
    tiempo_test_ms = toc()

    return y_validation, tiempo_entrenamiento_ms, tiempo_test_ms


def calular_acuracia(label_val, out_net):
    numero_acertos = 0
    cant_datos = len(label_val)

    for i in range(len(label_val)):
        if label_val[i] == out_net[i]:
            numero_acertos = numero_acertos + 1

    acuracia = (numero_acertos / cant_datos) * 100

    return acuracia


def calculo_desvio_padrao(input_vector):
    sumatoria = 0
    numero_de_elementos = len(input_vector)
    for i in range(numero_de_elementos):
        sumatoria = sumatoria + input_vector[i]

    media = sumatoria / numero_de_elementos
    sumatoria = 0
    for i in range(numero_de_elementos):
        sumatoria = + (input_vector[i] - media) ** 2
    desvio_padrao = math.sqrt(sumatoria / numero_de_elementos)

    return [media, desvio_padrao]


def calcular_matrix_de_confusion(labels_de_test,
                                 salida_predecida_por_la_red,
                                 titulo,
                                 enableDebug):
    global numero_de_grupos
    # plotMatrizDeConfusion = False

    matriz_de_confusion = np.zeros((numero_de_grupos, numero_de_grupos), dtype=int)

    for i in range(len(labels_de_test)):
        actual = int(labels_de_test[i]) - 1
        predicted = int(salida_predecida_por_la_red[i]) - 1

        matriz_de_confusion[actual][predicted] = matriz_de_confusion[actual][predicted] + 1

    if enableDebug:
        print(titulo, "[actual][predicted]", "\n", matriz_de_confusion)

    return matriz_de_confusion


def plotar_resultados(x_vector,
                      y_vector,
                      desvio_padrao_vector,
                      titulo,
                      nombre_curva,
                      x_label,
                      y_label,
                      ruta="figura.png"):
    plt.figure()
    plt.errorbar(x_vector, y_vector, yerr=desvio_padrao_vector, fmt='o', label=nombre_curva, capsize=5, ecolor='red')

    plt.legend(loc="best")
    plt.title(titulo)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(ruta, dpi=300, bbox_inches='tight')


def select_best_beam(enableDebug=False):
    global numero_de_grupos

    print('\n Reading pre-processed data ...')
    coord_input_train, coord_input_validation = read_inputs(debug=enableDebug)  # coordenadas
    coord_label_train, coord_label_validation = read_labels_data(debug=enableDebug)

    x = coord_input_validation[:, 0]
    y = coord_input_validation[:, 1]
    z = coord_input_validation[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=z, cmap='coolwarm')
    plt.pause(1)


    # config parameters
    if enableDebug:
        numero_experimentos = 2
    else:
        numero_experimentos = 1

    path_result = "../../results/"

    vector_acuracia = []
    vector_time_test = []
    vector_time_train = []
    vector_matriz_confusion = []
    coord_prediction =[]
    matriz_confusion_sumatoria = np.zeros((numero_de_grupos, numero_de_grupos), dtype=float)

    for i in range(numero_experimentos):  # For encargado de ejecutar el numero de rodadas (experimentos)
        print("\n\n >> Experimento: " + str(i))

        coord_prediction, time_train, time_test = neural_network(coord_input_train,
                                                                 coord_label_train,
                                                                 coord_input_validation)

        # #----------------- CALCULA MATRIZ DE CONFUSION -----------------------
        titulo = "Matriz_Confucao_" + str(i)

        if enableDebug:
            print("coord_label_validation = \n", coord_label_validation)
            print("coord_prediction = \n", coord_prediction)

        matriz_de_confusion = calcular_matrix_de_confusion(coord_label_validation,
                                                           coord_prediction,
                                                           titulo,
                                                           enableDebug)

        matriz_confusion_sumatoria = matriz_confusion_sumatoria + matriz_de_confusion
        vector_matriz_confusion.append(matriz_de_confusion)

        acuracia = calular_acuracia(coord_label_validation, coord_prediction)
        vector_acuracia.append(acuracia)
        vector_time_train.append(time_train)
        vector_time_test.append(time_test)

    # ----------------- CALCULA ESTADISTICAS -----------------------
    [acuracia_media, acuracia_desvio_padrao] = calculo_desvio_padrao(vector_acuracia)
    [time_train_media, time_train_desvio_padrao] = calculo_desvio_padrao(vector_time_train)
    [time_test_media, time_test_desvio_padrao] = calculo_desvio_padrao(vector_time_test)
    matriz_confusion_media = matriz_confusion_sumatoria / numero_experimentos

    # ----------------- IMPRIME MATRIZ DE CONFUSION MEDIA -----------------------
    titulo_mc = "** MATRIZ DE CONFUSÃO MÉDIA **"
    titulo_archivo = "matrix_de_confucion"
    print("\nAcuracia media = {:.2f}%".format(acuracia_media)
          + ";  dp = {:.2f}%".format(acuracia_desvio_padrao) +
          "\nTempo de entrenamento medio = {:.2f}ms".format(time_train_media * 1000) +
          ";  dp = {:.2f}ms".format(time_train_desvio_padrao * 1000) +
          "\nTempo de predição medio = {:.2f}ms".format(time_test_media * 1000) +
          ";  dp = {:.2f}ms".format(time_test_desvio_padrao * 1000))

    df_cm = pd.DataFrame(matriz_confusion_media, index=range(1, numero_de_grupos + 1),
                         columns=range(1, numero_de_grupos + 1))
    path_confusion_matriz = path_result + 'confusionMatrix/' + titulo_archivo + ".png"
    if enableDebug:
        print("matriz de confução media [actual][predicted]= \n", df_cm)
    pretty.pretty_plot_confusion_matrix(df_cm, cmap='Blues', title=titulo_mc, nombreFigura=path_confusion_matriz,
                                        pred_val_axis='y')

    return coord_input_train, coord_input_validation, coord_label_train, coord_label_validation, coord_prediction, df_cm


def create_keras_model(numero_de_salidas):
    local_model = tf.keras.models.Sequential()
    # model.add(tf.keras.layers.Flatten(input_shape=(5750,))
    local_model.add(tf.keras.layers.Dense(3, activation=tf.nn.relu))
    local_model.add(tf.keras.layers.Dense(numero_de_salidas + 1, activation=tf.nn.softmax))

    local_model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])

    return local_model


# ------------------ MAIN -------------------#
if __name__ == '__main__':
    epocas = 5
    numero_de_antenas_por_grupo = 32
    numero_de_grupos = round(256 / numero_de_antenas_por_grupo)

    process_and_save_output_beams(numero_de_antenas_por_grupo)

    model = create_keras_model(numero_de_grupos)

    select_best_beam(enableDebug=False)
