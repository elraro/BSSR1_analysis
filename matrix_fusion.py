import numpy as np


def matrix_fusion(matrix_1_dict, matrix_2_dict):
    matrix_list = list()
    matrix_1 = matrix_1_dict["matrix"]
    matrix_2 = matrix_2_dict["matrix"]
    for alpha in np.arange(0, 1.01, 0.01):
        matrix_list.append(matrix_fusion_alpha(matrix_1, matrix_2, alpha))
    matrix_list_train = list()
    matrix_1_train = matrix_1_dict["matrix_train"]
    matrix_2_train = matrix_2_dict["matrix_train"]
    for alpha in np.arange(0, 1.01, 0.01):
        matrix_list_train.append(matrix_fusion_alpha(matrix_1_train, matrix_2_train, alpha))
    matrix_list_test = list()
    matrix_1_test = matrix_1_dict["matrix_test"]
    matrix_2_test = matrix_2_dict["matrix_test"]
    for alpha in np.arange(0, 1.01, 0.01):
        matrix_list_test.append(matrix_fusion_alpha(matrix_1_test, matrix_2_test, alpha))
    matrix_total = dict()
    matrix_total["matrix"] = matrix_list
    matrix_total["matrix_train"] = matrix_list_train
    matrix_total["matrix_test"] = matrix_list_test
    return matrix_total


def matrix_fusion_alpha(matrix_1, matrix_2, alpha):
    matrix_scores = matrix_1 * alpha + (1 - alpha) * matrix_2
    matrix_fus = dict()
    matrix_fus["matrix"] = matrix_scores
    matrix_fus["aux"] = alpha
    return matrix_fus
