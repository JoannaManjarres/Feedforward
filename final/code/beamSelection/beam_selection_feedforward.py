import csv
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from autokeras import StructuredDataClassifier
from sklearn.model_selection import train_test_split


from tensorflow.keras.layers import Flatten

from keras import initializers
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import top_k_accuracy_score
from sklearn.preprocessing import StandardScaler
from operator import itemgetter
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

    input_cache_file = np.load(path + "lidar_train_raymobtime.npz", allow_pickle=True)
    data_lidar_train = input_cache_file["input"].astype(int)

    input_cache_file = np.load(path + "lidar_validation_raymobtime.npz", allow_pickle=True)
    data_lidar_validation = input_cache_file["input"].astype(int)

    x_dimension = len (data_lidar_train[0, :, 0, 0])
    y_dimension = len (data_lidar_train[0, 0, :, 0])
    z_dimension = len (data_lidar_train[0, 0, 0, :])
    dimension_of_coordenadas = x_dimension * y_dimension * z_dimension
    number_of_samples_train = data_lidar_train.shape[0]

    all_data_train = np.zeros([number_of_samples_train, dimension_of_coordenadas], dtype=np.int8)
    a = np.zeros(dimension_of_coordenadas, dtype=np.int8)

    for i in range(number_of_samples_train):
        a = data_lidar_train[i, :, :, :].reshape(1, dimension_of_coordenadas)
        all_data_train[i] = a

    number_of_samples_validation = data_lidar_validation.shape[0]
    all_data_validation = np.zeros([number_of_samples_validation, dimension_of_coordenadas], dtype=np.int8)
    b = np.zeros(dimension_of_coordenadas, dtype=np.int8)

    for i in range(number_of_samples_validation):
        b = data_lidar_validation[i, :, :, :].reshape(1, dimension_of_coordenadas)
        all_data_validation[i] = b

    return all_data_train, all_data_validation, data_lidar_train, data_lidar_validation

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

def parameters_configuration_coord():
    using_standarScaler = False
    index_beam_train, index_beam_test = read_beams ()
    y_train = index_beam_train
    y_test = index_beam_test

    if using_standarScaler:
        x_train_1, x_test_1 = pre_process_coord (using_standarScaler=True)
        batch = 25
        neurons_input = 3
        neurons_hidden = 100
        activation_function_input = 'relu'
        activation_function_hidden = 'sigmoid'
        accuracy, all_y_predict = model_feedforward (x_train_1, x_test_1, y_train, y_test, batch, neurons_input,
                                                     neurons_hidden, activation_function_input,
                                                     activation_function_hidden)
        print('Usando como pre-processamento a normalizacao pelo standarScaler')
        print('Accuracy: ', accuracy)

    else:
        x_train_1, x_test_1 = pre_process_coord (using_standarScaler=False)

        #Formula camadas ocultas
        #Neuronios = (Neuronios de entrada + Neuronios de saida) / 2
        #Neuronios = (3 + 256) / 2 = 129.5
        #Neuronios = (129.5 + 256) / 2 = 192.75


        batch = 50
        neurons_input = 3
        neurons_hidden = 256
        activation_function_input = 'relu'
        activation_function_hidden = 'sigmoid'
        accuracy, all_y_predict = model_feedforward (x_train_1, x_test_1, y_train, y_test, batch, neurons_input,
                                                     neurons_hidden, activation_function_input,
                                                     activation_function_hidden)
        print('Usando como pre-processamento a normalizacao pelo maximo')
        print('Accuracy: ', accuracy)

def model_feedforward(x_train, x_test, y_train, y_test, batch, neurons_input, neurons_hidden, activation_function_input, activation_function_hidden):
    epocas = 200
    batch_size = batch #50 #128

    local_model = tf.keras.models.Sequential()

    local_model.add(tf.keras.layers.InputLayer(input_shape=(x_train.shape[1])))

    local_model.add(tf.keras.layers.Dense(units=neurons_input, activation=activation_function_input)) #'relu' 'sigmoid'
    #local_model.add (keras.layers.Dense(units=512, activation='sigmoid'))
    local_model.add(tf.keras.layers.Dense(units=neurons_hidden, activation=activation_function_hidden))
    local_model.add(tf.keras.layers.Dense(256, activation=tf.keras.activations.softmax)) #activation=tf.nn.log_softmax
    local_model.summary()
    local_model.compile (optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.01), #   0.0001
                         #optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                         #optimizer=tf.optimizers.RMSprop(learning_rate=0.01, rho=0.9),
                         #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='sum_over_batch_size'),
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(reduction='sum_over_batch_size'),
                         #loss=tf.keras.losses.CategoricalCrossentropy(),
                         #loss=tf.keras.losses.mean_squared_logarithmic_error(),
                         #loss = tf.keras.losses.MeanSquaredError(),
                         metrics=['accuracy'])
    history = local_model.fit (x_train,
                               y_train,
                               epochs=epocas,
                               batch_size=batch_size,
                               verbose=1, #1
                               validation_split=0.2,
                               validation_data=(x_test, y_test))

    y_predict = local_model.predict (x_test)
    y_predict_top_1 = np.argmax(y_predict, axis=-1)
    accuracy = accuracy_score(y_test, y_predict_top_1)
    print ('Accuracy: ', accuracy)

    #preparing data for top k
    y_predict_shorted = np.argsort(y_predict, axis=-1)
    y_predict_index_shorted = np.flip(y_predict_shorted, axis=1)

    top_5 = y_predict_index_shorted[:, :5]

    top_k = [1, 5, 10, 20, 30, 40, 50]

    acuracia = []

    for i in range (len (top_k)):
        acerto = 0
        nao_acerto = 0
        for amostra_a_avaliar in range (len (y_predict_index_shorted)):
            group = np.array (y_predict_index_shorted)[:, 0:top_k[i]]
            if (y_test[amostra_a_avaliar] in group[amostra_a_avaliar]):
                acerto = acerto + 1
            else:
                nao_acerto = nao_acerto + 1

        acuracia.append (acerto / len (y_predict_index_shorted))

    print('Top k', top_k)
    print('Acuracia', acuracia)

    name = 'feedforward_coord'
    df_acuracia_comite_top_k = pd.DataFrame (acuracia)
    path = '../../results/'
    df_acuracia_comite_top_k.to_csv (path + 'acuracia_' + name + '_top_k.csv')



    title_accuracy = 'model accuracy \n Feedforward: Coordenadas'
    title_loss = 'model loss \n Feedforward: Coordenadas'
    name = 'feedforward_coord'
    plot_model_evolution (title_accuracy, title_loss, history, name_figure=name)

    return accuracy, y_predict

def pre_process_coord(using_standarScaler):
    x_train_1, x_test_1 = read_coordinates()



    if using_standarScaler:
        # Normalizacao dos dados com StandardScaler
        sc = StandardScaler()
        coord_x_train = sc.fit_transform (x_train_1[:, 0].reshape (-1, 1))
        coord_y_train = sc.fit_transform (x_train_1[:, 1].reshape (-1, 1))
        coord_z_train = sc.fit_transform (x_train_1[:, 2].reshape (-1, 1))
        x_train = np.column_stack ((coord_x_train, coord_y_train, coord_z_train))

        coord_x_test = sc.fit_transform (x_test_1[:, 0].reshape (-1, 1))
        coord_y_test = sc.fit_transform (x_test_1[:, 1].reshape (-1, 1))
        coord_z_test = sc.fit_transform (x_test_1[:, 2].reshape (-1, 1))
        x_test = np.column_stack ((coord_x_test, coord_y_test, coord_z_test))

    else:
        # Normalizacao dos dados divido pelo maximo
        coord_x_train = x_train_1[:, 0] / x_train_1[:, 0].max ()
        coord_y_train = x_train_1[:, 1] / x_train_1[:, 1].max ()
        coord_z_train = x_train_1[:, 2] / x_train_1[:, 2].max ()
        x_train = np.column_stack ((coord_x_train, coord_y_train, coord_z_train))
        # x_train = np.column_stack ((coord_x_train, coord_y_train))

        coor_x_test = x_test_1[:, 0] / x_test_1[:, 0].max ()
        coor_y_test = x_test_1[:, 1] / x_test_1[:, 1].max ()
        coor_z_test = x_test_1[:, 2] / x_test_1[:, 2].max ()
        x_test = np.column_stack ((coor_x_test, coor_y_test, coor_z_test))
        # x_test = np.column_stack ((coor_x_test, coor_y_test))

    return x_train, x_test

def read_results_luan():
    path = '../../results/results_Luan/'
    filename = 'log_epochs-200_opt-adam.csv'

    results = pd.read_csv(path+filename)

    print(results.head())
    top_1=results['val_top1'][199]
    top_5=0.97
    top_10=results['val_top10'][199]
    top_20=results['val_top20'][199]
    top_30=results['val_top30'][199]
    top_40=results['val_top40'][199]
    top_50=results['val_top50'][199]

    acuracia=[top_1, top_5, top_10, top_20, top_30, top_40, top_50]
    top_k =['top_1', 'top_5','top_10', 'top_20', 'top_30', 'top_40', 'top_50']
    name = 'dnn_lidar'
    name_1 = 'dnn_lidar_with_names'
    df_acuracia_comite_top_k = pd.DataFrame (data=acuracia)
    df_acuracia_comite_top_k_1 = pd.DataFrame (data=acuracia, index=top_k)

    df_acuracia_comite_top_k.to_csv (path + 'acuracia_' + name + '_top_k.csv')
    df_acuracia_comite_top_k_1.to_csv (path + 'acuracia_' + name_1 + '_top_k.csv')

    results.plot(x='epoch', y=['accuracy', 'val_accuracy'])

    return results

def model_feedfoward_lidar(): #(batch, neurons_input, neurons_hidden, activation_function_input, activation_function_hidden):
    index_beam_train, index_beam_validation = read_beams()
    lidar_train, lidar_validation, _, _ = read_lidar()

    x_train = lidar_train
    x_validation = lidar_validation
    y_train = index_beam_train
    y_validation = index_beam_validation

    batch = 200 # [10, 50, 100, 128, 256]
    layers_hidden = 3
    normalization = True
    dropout_flag = True
    dropout = 0.3

    neurons_input = [512]
    neurons_hidden = [1024] #[512, 1024, 2048, 4096]   [32]#
    neurons_hidden_1 = [1024] #[32, 128, 256, 512, 1024, 2048, 4096]
    taxa_aprendizado = [0.00001]#[0.00001, 0.0001, 0.001, 0.01, 0.1]

    activation_function_input = 'LeakyReLU' #'relu' 'LeakyReLU' # ['relu', 'sigmoid']
    activation_function_hidden = ['LeakyReLU'] #['relu', 'sigmoid', 'LeakyReLU']
    activation_function_hidden_1 = ['LeakyReLU'] #['relu', 'sigmoid', 'LeakyReLU']


    if layers_hidden == 1:
        ## Unica camada oculta
        for i in range(len(neurons_hidden)):
        #for i in range (len (taxa_aprendizado)):
            epocas = 100
            batch_size = batch  # 50 #128

            local_model = tf.keras.models.Sequential()

            local_model.add (tf.keras.layers.InputLayer(input_shape=(x_train.shape [1])))
            local_model.add(tf.keras.layers.Dropout(rate=dropout))
            local_model.add (tf.keras.layers.Dense(units=neurons_input, activation=activation_function_input))  # 'relu' 'sigmoid'

            # local_model.add(tf.keras.layers. (0.6))

            local_model.add(tf.keras.layers.Dense(256, activation=tf.keras.activations.softmax))  # activation=tf.nn.log_softmax
            local_model.summary()
            local_model.compile(optimizer=tf.keras.optimizers.legacy.Adam (learning_rate=taxa_aprendizado[i]),
                                 # overfitting 0.001  0.009988
                                 loss=tf.keras.losses.SparseCategoricalCrossentropy (reduction='sum_over_batch_size'),
                                 metrics=['accuracy'])
            es = EarlyStopping(monitor='val_loss', patience=5)
            history = local_model.fit (x_train,
                                       y_train,
                                       epochs=epocas,
                                       batch_size=batch_size,
                                       verbose=1,  # 1
                                       validation_split=0.2)  # ,
            # validation_data=(x_validation, y_validation),
            # callbacks=[es])

            y_predict = local_model.predict (x_validation)
            y_predict_top_1 = np.argmax (y_predict, axis=-1)
            accuracy = accuracy_score (y_validation, y_predict_top_1)

            # y_predic_1 = local_model.predict_proba (x_test)

            # print('Acuracia: ',accuracy_score(y_test, y_predict))
            # print('Top 5: ',top_k_accuracy_score(y_test, local_model.predict (x_test), k=5))
            # print (classification_report (y_test.argmax (axis=1),
            #                              y_predict.argmax (axis=1)))
            # target_names=[str (x) for x in lb.classes_]))

            print ('Accuracy: ', accuracy)

            title_accuracy = 'model accuracy  \n Feedforward: Lidar'
            title_loss = 'model loss \n Feedforward: Lidar'
            name = 'feedforward_lidar_hidden_layers_' \
                   + str(layers_hidden) + \
                   '_neurons_input_' + str (neurons_input) + \
                   '_act_func_input_' + str (activation_function_input)+ \
                    '_taxa_aprendizado_' + str (taxa_aprendizado[i])+ \
                    'with_dropout' + str(dropout)

            plot_model_evolution(title_accuracy, title_loss, history, name_figure=name)

            # preparing data for top k
            y_predict_shorted = np.argsort (y_predict, axis=-1)
            y_predict_index_shorted = np.flip (y_predict_shorted, axis=1)

            calcular_top_k = False

            if calcular_top_k:
                top_5 = y_predict_index_shorted [:, :5]

                top_k = [1, 5, 10, 20, 30, 40, 50]

                acuracia = []

                for i in range (len (top_k)):
                    acerto = 0
                    nao_acerto = 0
                    for amostra_a_avaliar in range (len (y_predict_index_shorted)):
                        group = np.array (y_predict_index_shorted) [:, 0:top_k [i]]
                        if (y_validation [amostra_a_avaliar] in group [amostra_a_avaliar]):
                            acerto = acerto + 1
                        else:
                            nao_acerto = nao_acerto + 1

                    acuracia.append (acerto / len (y_predict_index_shorted))

                print ('Top k', top_k)
                print ('Acuracia', acuracia)

                name = 'feedforward_lidar'
                df_acuracia_comite_top_k = pd.DataFrame (acuracia)
                path = '../../results/'
                df_acuracia_comite_top_k.to_csv (path + 'acuracia_' + name + '_top_k.csv')

            # save model to file
        # model.save ('model.h5')

    elif layers_hidden == 2:
        ## VArias camadas ocultas
        for i in range(len(neurons_input)):
            for j in range(len(neurons_hidden)):
                for k in range(len(activation_function_hidden)):
                    for l in range(len(activation_function_hidden_1)):
                        epocas = 300
                        batch_size = batch  # 50 #128

                        local_model = tf.keras.models.Sequential()

                        local_model.add(tf.keras.layers.InputLayer(input_shape=(x_train.shape[1])))
                        local_model.add(tf.keras.layers.Dropout(rate=dropout))
                        local_model.add(tf.keras.layers.BatchNormalization())

                        local_model.add(tf.keras.layers.Dense(units=neurons_input[i], activation=activation_function_input))  # 'relu' 'sigmoid'
                        local_model.add(tf.keras.layers.Dropout(rate=dropout))
                        local_model.add (tf.keras.layers.BatchNormalization())

                        local_model.add(tf.keras.layers.Dense(units=neurons_hidden[j], activation=activation_function_hidden[k]))
                        local_model.add(tf.keras.layers.Dropout(rate=dropout))
                        local_model.add (tf.keras.layers.BatchNormalization())

                        #local_model.add(tf.keras.layers.Dense(units=neurons_hidden_1[l], activation=activation_function_hidden_1[l]))
                        #local_model.add(tf.keras.layers.Dropout(rate=dropout))

                        #local_model.add (keras.layers.Dense(units=1024, activation='sigmoid'))
                        #local_model.add (tf.keras.layers.Dense(units=neurons_hidden, activation=activation_function_hidden))
                        local_model.add(tf.keras.layers.Dense(256, activation=tf.keras.activations.softmax))  # activation=tf.nn.log_softmax
                        local_model.summary()
                        local_model.compile(optimizer=tf.keras.optimizers.legacy.Adam (learning_rate=0.00001),  # overfitting 0.001  0.009988
                                             loss=tf.keras.losses.SparseCategoricalCrossentropy(reduction='sum_over_batch_size'),
                                             metrics=['accuracy'])
                        es = EarlyStopping (monitor='val_loss', patience=5)
                        history = local_model.fit (x_train,
                                                   y_train,
                                                   epochs=epocas,
                                                   batch_size=batch_size,
                                                   verbose=1,  # 1
                                                   validation_split=0.2)#,
                                                   #validation_data=(x_validation, y_validation),
                                                   #callbacks=[es])

                        y_predict = local_model.predict(x_validation)
                        y_predict_top_1 = np.argmax(y_predict, axis=-1)
                        accuracy = accuracy_score(y_validation, y_predict_top_1)

                        # y_predic_1 = local_model.predict_proba (x_test)

                        # print('Acuracia: ',accuracy_score(y_test, y_predict))
                        # print('Top 5: ',top_k_accuracy_score(y_test, local_model.predict (x_test), k=5))
                        # print (classification_report (y_test.argmax (axis=1),
                        #                              y_predict.argmax (axis=1)))
                        # target_names=[str (x) for x in lb.classes_]))

                        print ('Accuracy: ', accuracy)

                        title_accuracy = 'model accuracy with Dropout and batch normalization \n Feedforward: Lidar'
                        title_loss = 'model loss with Dropout \n Feedforward: Lidar'
                        path = '../../results/feedforward/Dropout/2_camadas/batch_normalization/'
                        name = 'feedforward_lidar_Dropout_hidden_layers_'\
                               +str(layers_hidden)+\
                               '_neurons_input_'+str(neurons_input[i])+\
                               '_act_func_input_'+str(activation_function_input)+\
                               '_neurons_hidden_'+str(neurons_hidden[j])+\
                               '_act_func_hidden_'+str(activation_function_hidden[k])
                        #+ \
                        #       '_neurons_hidden_1_'+str(neurons_hidden_1[l])+ \
                        #       '_act_func_hidden_' + '_act_func_hidden_1'+str(activation_function_hidden_1[l])+'dropout_'+str(dropout))

                        plot_model_evolution(title_accuracy, title_loss, history, name_figure=name, path_1=path)

                        # preparing data for top k
                        y_predict_shorted = np.argsort (y_predict, axis=-1)
                        y_predict_index_shorted = np.flip (y_predict_shorted, axis=1)

                        calcular_top_k = False

                        if calcular_top_k:
                            top_5 = y_predict_index_shorted[:, :5]

                            top_k = [1, 5, 10, 20, 30, 40, 50]

                            acuracia = []

                            for i in range (len(top_k)):
                                acerto = 0
                                nao_acerto = 0
                                for amostra_a_avaliar in range (len(y_predict_index_shorted)):
                                    group = np.array (y_predict_index_shorted)[:, 0:top_k[i]]
                                    if (y_validation[amostra_a_avaliar] in group[amostra_a_avaliar]):
                                        acerto = acerto + 1
                                    else:
                                        nao_acerto = nao_acerto + 1

                                acuracia.append (acerto / len (y_predict_index_shorted))

                            print ('Top k', top_k)
                            print ('Acuracia', acuracia)

                            name = 'feedforward_lidar_with_Dropout_hidden_layers_'
                            df_acuracia_comite_top_k = pd.DataFrame(acuracia)
                            path = '../../results/feedforward/Dropout/2_camadas/'
                            df_acuracia_comite_top_k.to_csv (path + 'acuracia_' + name + '_top_k.csv')


                        # save model to file
                        #model.save ('model.h5')

    elif layers_hidden == 3:
        ## VArias camadas ocultas
        for i in range(len(neurons_input)):
            for j in range(len(neurons_hidden)):
                for m in range(len(neurons_hidden_1)):
                    for k in range(len(activation_function_hidden)):
                        for l in range(len(activation_function_hidden_1)):
                            epocas = 300
                            batch_size = batch

                            local_model = tf.keras.models.Sequential()

                            # Camada de entrada
                            local_model.add(tf.keras.layers.InputLayer(input_shape=(x_train.shape[1])))
                            if dropout_flag:
                                local_model.add(tf.keras.layers.Dropout(rate=dropout))
                            if normalization:
                                local_model.add(tf.keras.layers.BatchNormalization())

                            # Camada oculta: 1
                            local_model.add(tf.keras.layers.Dense(units=neurons_input[i], activation=activation_function_input))  # 'relu' 'sigmoid'
                            if dropout_flag:
                                local_model.add(tf.keras.layers.Dropout(rate=dropout))
                            if normalization:
                                local_model.add(tf.keras.layers.BatchNormalization())

                            # Camada oculta: 2
                            local_model.add(tf.keras.layers.Dense(units=neurons_hidden[j], activation=activation_function_hidden[k]))
                            if dropout_flag:
                                local_model.add(tf.keras.layers.Dropout(rate=dropout))
                            if normalization:
                                local_model.add(tf.keras.layers.BatchNormalization())

                            # Camada oculta: 3
                            local_model.add(tf.keras.layers.Dense(units=neurons_hidden_1[m], activation=activation_function_hidden_1[l]))
                            if dropout_flag:
                                local_model.add(tf.keras.layers.Dropout(rate=dropout))
                            if normalization:
                                local_model.add (tf.keras.layers.BatchNormalization())

                            # Camada de saida
                            local_model.add(tf.keras.layers.Dense(256, activation=tf.keras.activations.softmax))  # activation=tf.nn.log_softmax
                            local_model.summary()
                            local_model.compile(optimizer=tf.keras.optimizers.legacy.Adam (learning_rate=0.00001),  # overfitting 0.001  0.009988
                                                 loss=tf.keras.losses.SparseCategoricalCrossentropy(reduction='sum_over_batch_size'),
                                                 metrics=['accuracy'])
                            es = EarlyStopping(monitor='val_loss', patience=5)
                            history = local_model.fit(x_train,
                                                       y_train,
                                                       epochs=epocas,
                                                       batch_size=batch_size,
                                                       verbose=1,  # 1
                                                       validation_split=0.2)

                            y_predict = local_model.predict(x_validation)
                            y_predict_top_1 = np.argmax(y_predict, axis=-1)
                            accuracy = accuracy_score(y_validation, y_predict_top_1)

                            print('Accuracy: ', accuracy)

                            if dropout_flag:
                                title_accuracy = 'model accuracy with Dropout \n Feedforward: Lidar'
                                title_loss = 'model loss with Dropout \n Feedforward: Lidar'
                                path = '../../results/feedforward/'+ str (layers_hidden) +'_hidden_layers/dropout/rate_' + str(dropout) + '/'
                                name = 'feedforward_lidar_Dropout_hidden_layers_' \
                                       + str (layers_hidden) + \
                                       '_neu_input_' + str (neurons_input [i]) + \
                                       '_func_input_' + str (activation_function_input) + \
                                       '_neu_hidden_' + str (neurons_hidden [j]) + \
                                       '_func_hidden_' + str (activation_function_hidden [k]) + \
                                       '_neu_hidden_1_' + str (neurons_hidden_1 [m]) + \
                                       '_func_hidden_1_' + str (activation_function_hidden_1 [l])
                            if normalization:
                                title_accuracy = 'model accuracy with Dropout and batch normalization \n Feedforward: Lidar'
                                title_loss = 'model loss with Dropout and batch normalization \n Feedforward: Lidar'
                                path = '../../results/feedforward/'+ str (layers_hidden) +'_hidden_layers/dropout/rate_' + str(dropout) + '/' + 'batch_normalization/'
                                name = 'feedforward_lidar_Dropout_Batch_hidden_layers_' \
                                       + str (layers_hidden) + \
                                       '_neu_input_' + str (neurons_input [i]) + \
                                       '_func_input_' + str (activation_function_input) + \
                                       '_neu_hidden_' + str (neurons_hidden [j]) + \
                                       '_func_hidden_' + str (activation_function_hidden [k]) + \
                                       '_neu_hidden_1_' + str (neurons_hidden_1 [m]) + \
                                       '_func_hidden_1_' + str (activation_function_hidden_1 [l])
                            if dropout_flag == False and normalization == False:
                                title_accuracy = 'model accuracy \n Feedforward: Lidar'
                                title_loss = 'model loss \n Feedforward: Lidar'
                                path = '../../results/feedforward/' + str(
                                    layers_hidden) + '_hidden_layers/'
                                name = 'feedforward_lidar_Dropout_hidden_layers_' \
                                       + str (layers_hidden) + \
                                       '_neu_input_' + str (neurons_input [i]) + \
                                       '_func_input_' + str (activation_function_input) + \
                                       '_neu_hidden_' + str (neurons_hidden [j]) + \
                                       '_func_hidden_' + str (activation_function_hidden [k]) + \
                                       '_neu_hidden_1_' + str (neurons_hidden_1 [m]) + \
                                       '_func_hidden_1_' + str (activation_function_hidden_1 [l])
                            plot_model_evolution(title_accuracy, title_loss, history, name_figure=name, path_1=path)

                            # preparing data for top k
                            y_predict_shorted = np.argsort (y_predict, axis=-1)
                            y_predict_index_shorted = np.flip (y_predict_shorted, axis=1)

                            calcular_top_k = False

                            if calcular_top_k:
                                top_5 = y_predict_index_shorted[:, :5]

                                top_k = [1, 5, 10, 20, 30, 40, 50]

                                acuracia = []

                                for i in range (len(top_k)):
                                    acerto = 0
                                    nao_acerto = 0
                                    for amostra_a_avaliar in range (len(y_predict_index_shorted)):
                                        group = np.array (y_predict_index_shorted)[:, 0:top_k[i]]
                                        if (y_validation[amostra_a_avaliar] in group[amostra_a_avaliar]):
                                            acerto = acerto + 1
                                        else:
                                            nao_acerto = nao_acerto + 1

                                    acuracia.append(acerto / len(y_predict_index_shorted))

                                print ('Top k', top_k)
                                print ('Acuracia', acuracia)

                                name = 'feedforward_lidar_3_hidden_layers_'
                                df_acuracia_comite_top_k = pd.DataFrame(acuracia)
                                df_acuracia_comite_top_k.to_csv(path + 'acuracia_' + name + '_top_k.csv')

    return accuracy, y_predict

def plot_model_evolution(title_accuracy, title_loss, history, name_figure, path_1):
    #path = '../../results/feedforward/'
    path = path_1

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(title_accuracy)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='best')
    plt.savefig(path + 'acc'+'_'+name_figure +'.png')
    #plt.show()
    plt.clf()

    #plt.figure()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(title_loss)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='best')
    plt.savefig (path + 'loss' + '_' + name_figure + '.png')
    #plt.show()
    plt.clf ()
    #plt.close(fig=name_figure)

def model_convolutional():
    print (tf.__version__)
    _, _, all_data_train, all_data_test = read_lidar()
    index_beam_train, index_beam_test = read_beams()

    cnn_neural_network = tf.keras.models.Sequential()
    cnn_neural_network.add(tf.keras.layers.Conv3D(5, (3, 3, 3), activation='relu', input_shape=(20, 200, 10, 1)))
    cnn_neural_network.add(tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)))
    #cnn_neural_network.add(tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_uniform'))
    #cnn_neural_network.add(tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)))
    cnn_neural_network.add(tf.keras.layers.Flatten())
    cnn_neural_network.add(tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_uniform'))
    cnn_neural_network.add(tf.keras.layers.Dense(256, activation='softmax'))

    cnn_neural_network.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(reduction='sum_over_batch_size'),
                               optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
                               metrics=['accuracy'])
    history = cnn_neural_network.fit(all_data_train, index_beam_train, batch_size=100, epochs=50, verbose=1, validation_split=0.2)
    score = cnn_neural_network.evaluate(all_data_test, index_beam_test, verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

    title_accuracy = 'model accuracy  \n CNN: Lidar'
    title_loss = 'model loss \n CNN: Lidar'
    name = 'CNN_lidar'

    plot_model_evolution(title_accuracy=title_accuracy,
                         title_loss=title_loss,
                         history=history,
                         name_figure=name)

    a=0


def auto_model():
    all_data_train, all_data_test, _, _ = read_lidar()
    index_beam_train, index_beam_test = read_beams()

    X_train, X_test, y_train, y_test = train_test_split (all_data_train, index_beam_train, test_size=0.2, random_state=1)
    print (X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    clf = StructuredDataClassifier(max_trials=3)
    clf.fit(X_train, y_train)
    loss, acc = clf.evaluate(X_test, y_test)
    print('Accuracy: %3f' % acc)

    index_beam_predict = clf.predict(all_data_test)
    print ('Predicted: %.3f' % index_beam_predict[0])
    # get the best performing model
    model = clf.export_model ()
    # summarize the loaded model
    model.summary()



    a=0



#read_results_luan()
#parameters_configuration_coord()
model_feedfoward_lidar()

#auto_model()
#model_convolutional()
#print(tf.__version__)

#parameters_definition()