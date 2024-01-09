import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def read_lidar():
    path = "../../data/processed/lidar/"
    input_cache_file = np.load(path + "lidar_train_raymobtime.npz", allow_pickle=True)
    all_lidar_train = input_cache_file["input"]

    input_cache_file = np.load(path + "lidar_validation_raymobtime.npz", allow_pickle=True)
    all_lidar_test = input_cache_file["input"]

    return all_lidar_train, all_lidar_test

def read_beam():
    path = "../../data/processed/beams/"

    input_cache_file = np.load (path + "index_beams_combined_train.npz", allow_pickle=True)
    index_beam_train = input_cache_file["all_beam_combined_train"].astype(int)

    input_cache_file = np.load (path + "index_beams_combined_test.npz", allow_pickle=True)
    index_beam_test = input_cache_file["all_beam_combined_test"].astype(int)

    return index_beam_train, index_beam_test


def conv_model(input_train, input_validation, label_train, label_validation):

    shape_of_input = input_train[0].shape
    learning_rate = 0.0001
    batch_size =100
    no_epochs =50
    verbosity = 1
    validation_split = 0.2

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=(20,200,10,1)))

    model.add (tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_uniform'))

    model.add (tf.keras.layers.MaxPooling3D (pool_size=(2, 2, 2)))
    model.add (tf.keras.layers.Flatten())
    model.add (tf.keras.layers.Dense (256, activation='relu', kernel_initializer='he_uniform'))
    model.add (tf.keras.layers.Dense (256, activation='softmax'))

    # Compile the model
    model.compile (loss=tf.keras.losses.SparseCategoricalCrossentropy,
                   optimizer=tf.keras.optimizers.legacy.Adam(lr=learning_rate),
                   metrics=['accuracy'])

    # Fit data to model
    history = model.fit (input_train, label_train,
                         batch_size=batch_size,
                         epochs=no_epochs,
                         verbose=verbosity,
                         validation_split=validation_split)

    score = model.evaluate (input_validation, label_validation, verbose=0)
    print (f'Test loss: {score[0]} / Test accuracy: {score[1]}')

    # Plot history: Categorical crossentropy & Accuracy
    plt.plot (history.history['loss'], label='Categorical crossentropy (training data)')
    plt.plot (history.history['val_loss'], label='Categorical crossentropy (validation data)')
    plt.plot (history.history['accuracy'], label='Accuracy (training data)')
    plt.plot (history.history['val_accuracy'], label='Accuracy (validation data)')
    plt.title ('Model performance for Beam Selection with LiDAR')
    plt.ylabel ('Loss value')
    plt.xlabel ('No. epoch')
    plt.legend (loc="upper left")
    plt.show ()

    return model


lidar_train, lidar_validation = read_lidar()
index_beam_train, index_beam_validation = read_beam()

conv_model(lidar_train, lidar_validation, index_beam_train, index_beam_validation)
p=0