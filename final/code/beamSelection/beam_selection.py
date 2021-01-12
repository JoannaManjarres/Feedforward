#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 10:45:28 2020

@author: joanna
"""
import timeit
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import os
import shutil
from sklearn.preprocessing import StandardScaler
import confusion_matrix_pretty_print as pretty
from keras import initializers

sys.path.append('../preProcessing/')
from pre_processing_baseline_output import process_and_save_output_beams
from my_decision_regions import plot_decision_regions


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


def trainning(x_train,
              y_train,
              id):
    global model
    global epocas
    global batch_size
    global history

    tic()
    # train using the input data

    # log_dir = "results/log/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "../../results/logs/fit/experiment_" + str(id)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(x_train, y_train,
                        epochs=epocas,
                        batch_size=batch_size,
                        verbose=1,
                        callbacks=[tensorboard_callback],
                        validation_split=0.2)
    return toc()


def predict(x_validation):
    global model

    tic()
    y_validation = np.argmax(model.predict(x_validation), axis=-1)

    return y_validation, toc()


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
                                 enable_debug_flag):
    global numero_de_grupos

    matriz_de_confusion = np.zeros((numero_de_grupos, numero_de_grupos), dtype=int)

    for i in range(len(labels_de_test)):
        actual = int(labels_de_test[i]) - 1
        predicted = int(salida_predecida_por_la_red[i]) - 1

        matriz_de_confusion[actual][predicted] = matriz_de_confusion[actual][predicted] + 1

    if enable_debug_flag:
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


def select_best_beam(num_neuronios_entrada, enable_debug=False):
    global numero_de_grupos
    global x_train
    global x_test
    global y_train
    global y_test
    global model
    global history
    global numero_experimentos
    global epocas

    # config parameters
    if enable_debug:
        numero_experimentos = 2
        epocas = 4

    path_result = "../../results/"

    vector_acuracia = []
    acuracia_max = -100.0
    vector_time_test = []
    vector_time_train = []
    vector_matriz_confusion = []
    matriz_confusion_sumatoria = np.zeros((numero_de_grupos, numero_de_grupos), dtype=float)
    best_model = model
    best_history = history

    for i in range(numero_experimentos):  # For encargado de ejecutar el numero de rodadas (experimentos)
        print("\n\n >> Experimento: " + str(i))

        tf.keras.backend.clear_session()
        model = create_keras_model(numero_de_grupos,num_neuronios_entrada)
        time_train = trainning(x_train, y_train, i)

        coord_prediction, time_test = predict(x_test)

        if enable_debug:
            print("coord_label_validation = \n", y_test)
            print("coord_prediction = \n", coord_prediction)

        # #----------------- CALCULA MATRIZ DE CONFUSION -----------------------
        titulo = "Matriz_Confucao_" + str(i)
        matriz_de_confusion = calcular_matrix_de_confusion(y_test, coord_prediction, titulo, enable_debug)

        matriz_confusion_sumatoria = matriz_confusion_sumatoria + matriz_de_confusion
        vector_matriz_confusion.append(matriz_de_confusion)

        acuracia = calular_acuracia(y_test, coord_prediction)
        vector_acuracia.append(acuracia)
        vector_time_train.append(time_train)
        vector_time_test.append(time_test)

        if acuracia > acuracia_max:
            acuracia_max = acuracia
            best_model = model
            best_history = history

    model = best_model
    history = best_history

    # ----------------- CALCULA ESTADISTICAS -----------------------
    [acuracia_media, acuracia_desvio_padrao] = calculo_desvio_padrao(vector_acuracia)
    [time_train_media, time_train_desvio_padrao] = calculo_desvio_padrao(vector_time_train)
    [time_test_media, time_test_desvio_padrao] = calculo_desvio_padrao(vector_time_test)
    matriz_confusion_media = matriz_confusion_sumatoria / numero_experimentos
    # plotar_resultados(acuracia_media,,acuracia_desvio_padrao,"Acuracia", "acuracia1", "Experimentos","acuracia","../../results/accuracy" )


    # ----------------- IMPRIME MATRIZ DE CONFUSION MEDIA -----------------------
    titulo_mc = "** MATRIZ DE CONFUSÃO MÉDIA **"
    titulo_archivo = "matrix_de_confucion"
    path_confusion_matriz = path_result + 'confusionMatrix/' + titulo_archivo + ".png"
    imprimir_matriz_de_confucion(matriz_confusion_media, numero_de_grupos, path_confusion_matriz,
                                 titulo_mc)
    print("\nAcuracia media = {:.2f}%".format(acuracia_media)
          + ";  dp = {:.2f}%".format(acuracia_desvio_padrao) +
          "\nTempo de entrenamento medio = {:.2f}ms".format(time_train_media * 1000) +
          ";  dp = {:.2f}ms".format(time_train_desvio_padrao * 1000) +
          "\nTempo de predição medio = {:.2f}ms".format(time_test_media * 1000) +
          ";  dp = {:.2f}ms".format(time_test_desvio_padrao * 1000))

    # ----------------- RESULTADOS PARA EL MEJOR MODELO -----------------------
    titulo_mc = "** Melhor Modelo **"
    titulo_archivo = "melhor_modelo"
    path_confusion_matriz = path_result + 'confusionMatrix/' + titulo_archivo + ".png"

    coord_prediction, time_test = predict(x_test)
    matriz_de_confusion = calcular_matrix_de_confusion(y_test,
                                                       coord_prediction,
                                                       titulo_archivo,
                                                       enable_debug)
    imprimir_matriz_de_confucion(matriz_de_confusion, numero_de_grupos, path_confusion_matriz,
                                 titulo_mc)


def plot_trainning_history():
    global history

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def imprimir_matriz_de_confucion(matriz_confusion, tamano, path_con_nombre_de_arvhivo,
                                 titulo_figura):
    df_cm = pd.DataFrame(matriz_confusion, index=range(1, tamano + 1),
                         columns=range(1, tamano + 1))
    pretty.pretty_plot_confusion_matrix(df_cm, cmap='Blues', title=titulo_figura,
                                        nombreFigura=path_con_nombre_de_arvhivo, pred_val_axis='y')


def create_keras_model(numero_de_salidas, num_neuronios_entrada):
    local_model = tf.keras.models.Sequential()
    # model.add(tf.keras.layers.Flatten(input_shape=(5750,))
    # local_model.add(tf.keras.layers.Dense(9, kernel_initializer=initializers.random_normal(mean=0,stddev=0.5),
    #                                       activation=tf.nn.tanh))
                                          # activation = tf.nn.relu))
    local_model.add(tf.keras.layers.Dense(num_neuronios_entrada, kernel_initializer=initializers.random_uniform(minval=-0.2,
                                                                                            maxval=0.2, seed=None),
                                          activation=tf.nn.tanh))
    # local_model.add(tf.keras.layers.Dense(numero_de_salidas + 1, activation=tf.nn.relu))
    local_model.add(tf.keras.layers.Dense(numero_de_salidas + 1, activation=tf.nn.log_softmax))

    local_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01),
                        # optimizer=tf.optimizers.RMSprop(learning_rate=0.01, rho=0.9)
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                           reduction='sum_over_batch_size'),
                        # loss=tf.keras.losses.mean_squared_logarithmic_error(),
                        metrics=['accuracy'])
    # local_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    return local_model


def plot_input_output_relationship():
    global x_test
    global y_test
    global x_train
    global y_train

    fig, axs = plt.subplots(1, 2)
    fig.suptitle('Input data')

    for i in range(2):
        if i == 0:
            coord_x = x_train[:, 0]
            coord_y = x_train[:, 1]
            beam_group = y_train
            ax = axs[0]
        else:
            coord_x = x_test[:, 0]
            coord_y = x_test[:, 1]
            beam_group = y_test
            ax = axs[1]

        data_frame_test = pd.DataFrame(dict(coord_X=coord_x, coord_y=coord_y, beam_group=beam_group))
        data_frame_test.head()

        sns.scatterplot(data=data_frame_test,
                        x="coord_X",
                        y="coord_y",
                        style="beam_group",
                        sizes=(200, 200),
                        size="beam_group",
                        palette="deep",
                        legend="full",
                        hue="beam_group",
                        ax=ax)
    plt.show()

    # h_true = tf.histogram_fixed_width(y_test, value_range=(1,9), nbins=20)
    # print(type(h_true))
    plt.style.use('ggplot')
    plt.hist(y_test, bins=20, rwidth=8, color='steelblue', alpha=0.5)
    plt.title("Labels")

    num_intervals = round(np.sqrt(len(x_train[:, 0])))
    rango = max(x_train[:, 0]) - min(x_train[:, 0])
    width_of_intervals = rango / num_intervals

    plt.show()
    plt.style.use('ggplot')
    plt.hist(x_train[:, 0], bins='scott', color='steelblue', alpha=0.5)
    plt.title("Coordenadas x")
    plt.show()

    plt.style.use('ggplot')
    plt.hist(x_train[:, 1], bins='scott', color='steelblue', alpha=0.5)
    plt.title("Coordenadas y")
    plt.show()


def pearsonr_2_d(x, y):
    """computes pearson correlation coefficient based on the equation above
       where x is a 1D and y a 2D array"""

    upper = np.sum((x - np.mean(x)) * (y - np.mean(y, axis=1)[:, None]), axis=1)
    lower = np.sqrt(np.sum(np.power(x - np.mean(x), 2)) * np.sum(np.power(y - np.mean(y, axis=1)[:, None], 2), axis=1))

    rho = upper / lower

    return rho


# ------------------ MAIN -------------------#
if __name__ == '__main__':
    numero_de_antenas_por_grupo = 32
    numero_de_grupos = round(256 / numero_de_antenas_por_grupo)
    epocas = 100
    batch_size = 100
    numero_experimentos = 2
    enableDebug = False
    enable_scale = True
    num_neuronios_entrada = 9

    print('\n pre-processing output data ...')
    process_and_save_output_beams(numero_de_antenas_por_grupo)

    print('\n Reading pre-processed data ...')
    x_train, x_test = read_inputs(debug=enableDebug)
    y_train, y_test = read_labels_data(debug=enableDebug)
    if enable_scale:
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

    # print('\n  input-output relationship ...')
    # PCC = pearsonr_2_d(np.transpose(y_train), np.transpose(x_train))
    # print("PCC = ", PCC)
    # plot_input_output_relationship()

    print('\n creating NN model ...')
    tf.keras.backend.clear_session()
    model = create_keras_model(numero_de_grupos,num_neuronios_entrada)
    history = 0

    print('\n Selecting beam groups...')
    log_path = "../../results/logs/fit/"
    shutil.rmtree(log_path, ignore_errors=True)
    select_best_beam(num_neuronios_entrada, enable_debug=enableDebug)

    pesos = model.get_weights()
    print(pesos)

    print('\n Ploting trainning results...')
    plot_trainning_history()

    plot_decision_regions(x_test, y_test, clf=model, legend=2)
    plt.show()

    print('\n Opening tensorboard...')
    print(log_path)
    # os.system('conda activate tf')
    os.system('tensorboard --logdir ' + log_path)
