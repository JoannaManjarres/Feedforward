import csv
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import initializers
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import top_k_accuracy_score
import pandas as pd


def read_coordinates():
    #path = "/Users/joannamanjarres/git/Feedforward/data/coordinates/"
    path="../../data/processed/coord_input/"
    filename = "CoordVehiclesRxPerScene_s008.csv"
    limit_ep_train = 1564

    with open(path+filename) as csvfile:
        reader = csv.DictReader (csvfile)
        number_of_rows = len(list (reader))

    all_info_coord_val = np.zeros([11194, 5], dtype=object)
    all_coord = np.zeros([11194, 4], dtype=float)

    with open(path+filename) as csvfile:
        reader = csv.DictReader (csvfile)
        cont = 0
        for row in reader:
            if row['Val'] == 'V':
                all_info_coord_val[cont] = int(row['EpisodeID']), float (row['x']), float (row['y']), float (row['z']), row['LOS']
                all_coord[cont] = int(row['EpisodeID']), float(row['x']), float(row['y']), float(row['z'])
                cont += 1

    data_train = all_coord[(all_info_coord_val[:, 0] < limit_ep_train + 1)]
    data_test = all_coord[(all_info_coord_val[:, 0] > limit_ep_train)]

    coord_train = data_train[:, [1, 2, 3]]
    coord_test = data_test[:, [1, 2, 3]]

    return coord_train, coord_test
def read_beams():

    #path = "/Users/joannamanjarres/git/Rede_Neural/data/beams/"
    path = "../../data/processed/beams/"

    input_cache_file = np.load (path + "index_beams_combined_train.npz", allow_pickle=True)
    index_beam_train = input_cache_file["all_beam_combined_train"].astype(int)

    input_cache_file = np.load (path + "index_beams_combined_test.npz", allow_pickle=True)
    index_beam_test = input_cache_file["all_beam_combined_test"].astype(int)

    return index_beam_train, index_beam_test

def read_lidar():
    path = "../../data/processed/lidar/"

    input_cache_file = np.load (path + "index_beams_combined_train.npz", allow_pickle=True)
    index_beam_train = input_cache_file["all_beam_combined_train"].astype (int)

    input_cache_file = np.load (path + "index_beams_combined_test.npz", allow_pickle=True)
    index_beam_test = input_cache_file["all_beam_combined_test"].astype (int)

    return index_beam_train, index_beam_test

def parameters_definition():

    batch = [10, 50, 100, 128, 256]
    neurons_input = [10, 25, 50, 128]
    neurons_hidden = [128, 256, 512, 1024]
    activation_function_input = ['relu', 'sigmoid']
    activation_function_hidden = ['relu', 'sigmoid']

    #batch = [10, 50]
    #neurons_input = [10]
    #neurons_hidden = [128]
    #activation_function_input = ['relu']
    #activation_function_hidden = ['sigmoid']

    x_train_1 , x_test_1 = read_coordinates()
    y_train, y_test = read_beams()

    data_to_save =[]

    a=0
    path = '../results/'
    for i in range(len(batch)):
        for j in range(len(neurons_input)):
            for k in range(len(neurons_hidden)):
                for l in range(len(activation_function_input)):
                    for m in range(len(activation_function_hidden)):
                        accuracy = model_feedforward(x_train_1, x_test_1, y_train, y_test, batch[i], neurons_input[j], neurons_hidden[k], activation_function_input[l], activation_function_hidden[m])
                        print('Accuracy: ', accuracy, 'Batch: ', batch[i], 'Neurons Input: ', neurons_input[j], 'Neurons Hidden: ', neurons_hidden[k], 'Activation Function Input: ', activation_function_input[l], 'Activation Function Hidden: ', activation_function_hidden[m])
                        data = [accuracy, batch[i], neurons_input[j], neurons_hidden[k], activation_function_input[l], activation_function_hidden[m]]
                        data_to_save.append(data)


    #model_feedforward(x_train_1, x_test_1, y_train, y_test, batch, neurons_input, neurons_hidden, activation_function_input, activation_function_hidden)

    accuracy_to_save = pd.DataFrame(data_to_save)
    accuracy_to_save.to_csv('accuracy_result_with_config.csv')
def model_feedforward(x_train_1, x_test_1, y_train, y_test, batch, neurons_input, neurons_hidden, activation_function_input, activation_function_hidden):
    epocas = 100
    batch_size = batch #50 #128

    coord_x_train = x_train_1[:, 0] / x_train_1[:, 0].max()
    coord_y_train = x_train_1[:, 1] / x_train_1[:, 1].max()
    coord_z_train = x_train_1[:, 2] / x_train_1[:, 2].max()
    x_train = np.column_stack ((coord_x_train, coord_y_train, coord_z_train))

    coor_x_test = x_test_1[:, 0] / x_test_1[:, 0].max()
    coor_y_test = x_test_1[:, 1] / x_test_1[:, 1].max()
    coor_z_test = x_test_1[:, 2] / x_test_1[:, 2].max()
    x_test = np.column_stack ((coor_x_test, coor_y_test, coor_z_test))

    local_model = tf.keras.models.Sequential()

    local_model.add(keras.layers.InputLayer(input_shape=(x_train.shape[1])))

    #local_model.add(keras.layers.Dense(units=neurons_input, activation=activation_function_input)) #'relu' 'sigmoid'
    #local_model.add (keras.layers.Dense(units=512, activation='sigmoid'))
    local_model.add(keras.layers.Dense(units=3, activation=activation_function_hidden))
    local_model.add(tf.keras.layers.Dense(256, activation=tf.keras.activations.softmax))# activation=tf.nn.log_softmax))
    local_model.summary()
    local_model.compile (optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.01),
                         #optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                         #optimizer=tf.optimizers.RMSprop(learning_rate=0.01, rho=0.9),
                         #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='sum_over_batch_size'),
                         loss=tf.keras.losses.CategoricalCrossentropy(),
                         #loss=tf.keras.losses.mean_squared_logarithmic_error(),

                         #loss = tf.keras.losses.MeanSquaredError(),
                         metrics=['accuracy'])
    history = local_model.fit (x_train,
                               y_train,
                               epochs=epocas,
                               batch_size=batch_size,
                               verbose=0, #1
                               validation_split=0.2,
                               validation_data=(x_test, y_test))

    y_predict = local_model.predict (x_test)
    y_predict_top_1 = np.argmax(y_predict, axis=-1)
    accuracy = accuracy_score(y_test, y_predict_top_1)

    #y_predic_1 = local_model.predict_proba (x_test)


    #print('Acuracia: ',accuracy_score(y_test, y_predict))
    #print('Top 5: ',top_k_accuracy_score(y_test, local_model.predict (x_test), k=5))
    #print (classification_report (y_test.argmax (axis=1),
    #                              y_predict.argmax (axis=1)))
                                 # target_names=[str (x) for x in lb.classes_]))

    plt.plot (history.history['accuracy'])
    plt.plot (history.history['val_accuracy'])
    plt.title ('model accuracy')
    plt.ylabel ('accuracy')
    plt.xlabel ('epoch')
    plt.legend (['training', 'validation'], loc='best')
    plt.show ()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel ('epoch')
    plt.legend (['training', 'validation'], loc='best')
    plt.show()

    return accuracy, y_predict



def model_feedfoward_lidar():
    read_beams()
    #re




coord_train, coord_test =read_coordinates()
index_beam_train, index_beam_test = read_beams()

#model_feedforward(coord_train, coord_test, index_beam_train, index_beam_test)

x_train_1= coord_train
x_test_1 = coord_test
y_train = index_beam_train
y_test = index_beam_test
batch = 200

neurons_input =10
neurons_hidden=25
activation_function_input='relu'
activation_function_hidden='sigmoid'

accuracy, all_y_predict = model_feedforward(x_train_1, x_test_1, y_train, y_test, batch, neurons_input, neurons_hidden, activation_function_input, activation_function_hidden)
parameters_definition()