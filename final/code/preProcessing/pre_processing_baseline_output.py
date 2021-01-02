#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 20:39:28 2020

@author: joanna
"""

import numpy as np


def beams_log_scale(y, threshold_below_max):
    y_shape = y.shape

    for i in range(0, y_shape[0]):
        this_outputs = y[i, :]
        log_out = 20 * np.log10(this_outputs + 1e-30)
        min_value = np.amax(log_out) - threshold_below_max
        zeroed_value_indices = log_out < min_value
        this_outputs[zeroed_value_indices] = 0
        this_outputs = this_outputs / sum(this_outputs)
        y[i, :] = this_outputs

    return y


def get_beam_output(output_file):
    threshold_below_max = 1

    print("Reading dataset...", output_file)
    output_cache_file = np.load(output_file)
    y_matrix = output_cache_file['output_classification']
    print("Forma yMatrix", y_matrix.shape)
    y_matrix = np.abs(y_matrix)
    y_matrix /= np.max(y_matrix)
    num_classes = y_matrix.shape[1] * y_matrix.shape[2]

    y = y_matrix.reshape(y_matrix.shape[0], num_classes)
    y = beams_log_scale(y, threshold_below_max)

    return y, num_classes


def generate_groups_beams(y, k):
    labels = []

    for i in range(y.shape[0]):
        valor = round(np.argmax(y[i, :], axis=0) / k)
        labels.append(valor)

    return np.asarray(labels)


# -------------------------- MAIN ------------------------------

def process_and_save_output_beams(k):
    if k == 0:
        print("k should be greater than zero")
    else:
        baseline_path = "../../data/processed/output_beam/baseline/"

        # train
        output_train_file = baseline_path + 'beams_output_train.npz'
        y_train, num_classes = get_beam_output(output_train_file)

        output_validation_file = baseline_path + 'beams_output_validation.npz'
        y_validation, _ = get_beam_output(output_validation_file)

        save_path = "../../data/processed/output_beam/"
        debug_path = "../../data/processed/output_beam/debug/"

        # train
        label_train = generate_groups_beams(y_train, k)
        debug_train = label_train[0:20]
        np.savez(save_path + 'beams_output_train' + '.npz', output_training=label_train)
        np.savez(debug_path + 'beams_output_train' + '.npz', output_training=debug_train)

        # test
        label_validation = generate_groups_beams(y_validation, k)
        debug_validation = label_validation[0:2]
        np.savez(save_path + 'beams_output_validation' + '.npz', output_test=label_validation)
        np.savez(debug_path + 'beams_output_validation' + '.npz', output_test=debug_validation)
