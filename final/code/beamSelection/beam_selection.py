#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 10:45:28 2020

@author: joanna
"""
import timeit
import sys
import csv
import math
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

    label_train1 = np.char.mod('%d', label_train)
    label_validation1 = np.char.mod('%d', label_validation)

    return label_train, label_validation, label_train1.tolist(), label_validation1.tolist()


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


def neural_network(data_train,
                   label_train,
                   x_validation):
    global model

    print('\n Training NN ...')
    tic()
    # train using the input data
    model.fit(data_train, label_train, epochs=5)

    tiempo_entrenamiento_ms = toc()

    print('\n Selecting Beams using NN ...')
    tic()
    # classify some data
    salida_de_la_red = np.argmax(model.predict(x_validation), axis=-1)
    tiempo_test_ms = toc()

    return salida_de_la_red, tiempo_entrenamiento_ms, tiempo_test_ms


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
                                 salida_de_red,
                                 titulo):
    global numero_de_grupos
    # plotMatrizDeConfusion = False

    matriz_de_confusion = np.zeros((numero_de_grupos, numero_de_grupos), dtype=int)
    # print("Datos Test: ", datos_de_test)
    # print("Salida de la red ", salida_de_red)
    for i in range(len(labels_de_test)):
        esperado = int(labels_de_test[i]) - 1
        obtenido = int(salida_de_red[i]) - 1
        # print("esperado = ",esperado)
        # print("obtenido = ",obtenido)

        matriz_de_confusion[obtenido][esperado] = matriz_de_confusion[obtenido][esperado] + 1
    # print(matriz_de_confusion)

    # path_result = "../../results/"
    # np.savetxt('confusionMatrix/'+str(nombre_del_experimento)+ \
    #   '/MC_'+nombre_del_experimento+str(numero_del_experimento)+ \
    #   '_'+str(numero_de_la_matriz) +'.csv', matriz_de_confusion_prueba, \
    #   delimiter=',', fmt='%d')

    # np.savetxt(path_result+nombre_arq_MC, matriz_de_confusion, delimiter=',', fmt='%d')
    # if(plotMatrizDeConfusion):
    #     df_cm = pd.DataFrame(matriz_de_confusion, index=range(0,10), columns=range(0,10))
    #     pretty.pretty_plot_confusion_matrix(df_cm, cmap='Blues',title=titulo)

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
    input_train, input_validation = read_inputs(debug=enableDebug)  # coordenadas
    label_train, label_validation, label_train1, label_validation1 = read_labels_data(debug=enableDebug)  #

    # config parameters
    if enableDebug:
        # address_size = [6,24,34,44,54,64]
        address_size = [28]
        numero_experimentos = 2
    else:
        address_size = [28]
        # numero_experimentos = 1
        # address_size = [6,12,18,24,28,34,38,44,48,54,58,64]
        numero_experimentos = 10

    vector_time_train_media = []
    vector_time_test_media = []
    vector_acuracia_media = []

    vector_acuracia_desvio_padrao = []
    vector_time_train_desvio_padrao = []
    vector_time_test_desvio_padrao = []

    path_result = "../../results/"

    for j in range(len(address_size)):  # For encargado de variar el tamano de la memoria

        vector_acuracia = []
        vector_time_test = []
        vector_time_train = []
        vector_matriz_confusion = []
        matriz_confusion_sumatoria = np.zeros((numero_de_grupos, numero_de_grupos), dtype=float)

        print("Tamanho memoria: " + str(address_size[j]))

        for i in range(numero_experimentos):  # For encargado de ejecutar el numero de rodadas (experimentos)
            print("   Experimento: " + str(i))

            # -----------------USA LA RED WIZARD -------------------
            out_red, time_train, time_test = neural_network(input_train,
                                                            label_train,
                                                            input_validation)

            vector_time_train.append(time_train)
            vector_time_test.append(time_test)

            # #----------------- CALCULA MATRIZ DE CONFUSION -----------------------
            titulo = "** MATRIZ DE CONFUSÃO " + str(i) + " **" + " \n Address Size " + str(address_size[j])
            # nombre_arq_MC = "MC_Address_Size_"+ str(address_size[j])

            print("label de salida", label_validation1)
            print("salida de la red", out_red)
            print("tipo de salida de la red", type(out_red))
            print("tamaño salida de la red", len(out_red))

            matrizdeconfusion = calcular_matrix_de_confusion(label_validation1,
                                                             out_red,
                                                             titulo)
            print("confution matrix", matrizdeconfusion)
            matriz_confusion_sumatoria = matriz_confusion_sumatoria + matrizdeconfusion
            # vector_matriz_confusion.append(matrizdeconfusion)

            print('\n Measuring output performance ...')
            acuracia = calular_acuracia(label_validation1, out_red)
            vector_acuracia.append(acuracia)

        # ----------------- CALCULA ESTADISTICAS -----------------------
        [acuracia_media, acuracia_desvio_padrao] = calculo_desvio_padrao(vector_acuracia)
        [time_train_media, time_train_desvio_padrao] = calculo_desvio_padrao(vector_time_train)
        [time_test_media, time_test_desvio_padrao] = calculo_desvio_padrao(vector_time_test)
        matriz_confusion_media = matriz_confusion_sumatoria / numero_experimentos

        # ----------------- GUARDA VECTORES DE ESTADISTICAS -----------------------
        vector_acuracia_media.append(acuracia_media)
        vector_acuracia_desvio_padrao.append(acuracia_desvio_padrao)

        vector_time_train_media.append(time_train_media)
        vector_time_train_desvio_padrao.append(time_train_desvio_padrao)

        vector_time_test_media.append(time_test_media)
        vector_time_test_desvio_padrao.append(time_test_desvio_padrao)

        # np.savez( path_result+"metricas.npz",
        #          matriz_confusao = vector_matriz_confusion)

        # ----------------- IMPRIME MATRIZ DE CONFUSION MEDIA -----------------------
        # titulo_mc = "** MATRIZ DE CONFUSÃO MÉDIA ** \n Address Size " + str(address_size[j])
        titulo_mc = "matrix" +str(address_size[j])
        df_cm = pd.DataFrame(matriz_confusion_media, index=range(0, numero_de_grupos),
                             columns=range(0, numero_de_grupos))
        path_confusion_matriz = path_result + 'confusionMatrix/' + titulo_mc + ".png"
        print("debug", df_cm)
        pretty.pretty_plot_confusion_matrix(df_cm, cmap='Blues', title=titulo_mc, nombreFigura=path_confusion_matriz)

    # ----------------- GUARDA EM CSV VECTORES DE ESTADISTICAS  -----------------------
    print('\n Saving results files ...')

    with open(path_result + 'accuracy/acuracia.csv', 'w') as f:
        writer_acuracy = csv.writer(f, delimiter='\t')
        writer_acuracy.writerows(zip(address_size, vector_acuracia_media, vector_acuracia_desvio_padrao))

    with open(path_result + 'processingTime/time_train.csv', 'w') as f:
        writer_time_train = csv.writer(f, delimiter='\t')
        writer_time_train.writerows(zip(address_size, vector_acuracia_media, vector_time_train_desvio_padrao))

    with open(path_result + 'processingTime/time_test.csv', 'w') as f:
        writer_time_test = csv.writer(f, delimiter='\t')
        writer_time_test.writerows(zip(address_size, vector_time_test_media, vector_time_test_desvio_padrao))

    # ----------------- PLOT DE RESULTADOS  ------------------------------
    # titulo ="% de treinamiento =" + str(porcentaje_entrenamiento) +", Threshold = "+str(threshold)
    titulo = "Test"
    nombre_curva = "Dado com desvio padrão"

    plotar_resultados(address_size,
                      vector_acuracia_media,
                      vector_acuracia_desvio_padrao,
                      titulo,
                      nombre_curva,
                      "Tamanho da memória",
                      "Acuracia Média (%)",
                      ruta=path_result + "/accuracy/acuracia.png")

    plotar_resultados(address_size,
                      vector_time_train_media,
                      vector_time_train_desvio_padrao,
                      titulo,
                      nombre_curva,
                      "Tamanho da memória",
                      "Tempo de treinamento Médio (s)",
                      ruta=path_result + "/processingTime/time_train.png")

    plotar_resultados(address_size,
                      vector_time_test_media,
                      vector_time_test_desvio_padrao,
                      titulo,
                      nombre_curva,
                      "Tamanho da memória",
                      "Tempo de Teste Médio (s)",
                      ruta=path_result + "/processingTime/time_test.png")

    return input_train, input_validation, label_train, label_validation, out_red, df_cm


def select_best_beam_debug(enableDebug=False):
    global numero_de_grupos

    print('\n Reading pre-processed data ...')
    input_train, input_validation = read_inputs(debug=enableDebug)  # coordenadas
    label_train, label_validation, label_train1, label_validation1 = read_labels_data(debug=enableDebug)  #

    # config parameters
    if enableDebug:
        # address_size = [6,24,34,44,54,64]
        address_size = [28]
        numero_experimentos = 2
    else:
        address_size = [28]
        # numero_experimentos = 1
        # address_size = [6,12,18,24,28,34,38,44,48,54,58,64]
        numero_experimentos = 10

    print("input train", input_train.shape)
    # -----------------USA LA RED WIZARD -------------------
    out_red, time_train, time_test = neural_network(input_train,
                                                    label_train,
                                                    input_validation)

    print("label de salida", label_validation)
    print("forma de label de salida", label_validation.shape)

    print("salida de la red", out_red)
    print("forma de la salida de la red", out_red.shape)

    print("forma del modelo", model.output_shape)

    df_cm = 0
    return input_train, input_validation, label_train, label_validation, out_red, df_cm


def create_keras_model(numero_de_salidas):
    model = tf.keras.Sequential([
        # tf.keras.layers.Flatten(input_shape=(5750,)),
        tf.keras.layers.Dense(1, activation='relu'),
        tf.keras.layers.Dense(numero_de_salidas)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # model = tf.keras.models.Sequential()
    # model.add(tf.keras.layers.Dense(5750, activation=tf.nn.relu))
    # model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(30, activation=tf.nn.relu))
    # model.add(tf.keras.layers.Dense(1, activation=tf.nn.softmax))
    # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


# ------------------ MAIN -------------------#
k = 26
process_and_save_output_beams(k)
numero_de_grupos = round(256 / k)

model = create_keras_model(k)

[input_train,
 input_validation,
 label_train,
 label_validation,
 out_red,
 matriz_confusion_media_dataframe] = select_best_beam(enableDebug=True)
