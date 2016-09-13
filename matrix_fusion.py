import numpy as np


def matrix_fusion(matrix_1_dict, matrix_2_dict):
    matrix_1 = matrix_1_dict["matrix"]
    matrix_2 = matrix_2_dict["matrix"]
    length = len(matrix_1)
    matrix_scores = np.empty((length, length))
    for alpha in np.arange(0.01, 1, 0.01):
        scores = list()
        for i_1 in np.arange(0, length, 1):
            row_1 = matrix_1[:,i_1]
            for cell_1 in row_1:
                for i_2 in np.arange(0, length, 1):
                    row_2 = matrix_2[:, i_2]
                    # tengo que usar .flat
                    for cell_2 in row_2.flat:
                        scores.append(cell_1 * alpha + (1 - alpha) * cell_2)
                    matrix_scores[i_1] = np.asarray(scores)