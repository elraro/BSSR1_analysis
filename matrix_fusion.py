import numpy as np


def matrix_fusion(matrix_1_dict, matrix_2_dict):
    matrix_list = list()
    matrix_1 = matrix_1_dict["matrix"]
    matrix_2 = matrix_2_dict["matrix"]
    for alpha in np.arange(0, 1.01, 0.01):
        matrix_list.append(matrix_fusion_alpha(matrix_1, matrix_2, alpha))
    return matrix_list


def matrix_fusion_alpha(matrix_1, matrix_2, alpha):
    matrix_scores = matrix_1 * alpha + (1 - alpha) * matrix_2
    matrix_fus = dict()
    matrix_fus["matrix"] = matrix_scores
    matrix_fus["aux"] = alpha
    return matrix_fus
