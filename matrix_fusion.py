import numpy as np

def matrix_fusion(matrix_1_dict, matrix_2_dict):
    matrix_1 = matrix_1_dict["matrix"]
    matrix_2 = matrix_2_dict["matrix"]
    length = len(matrix_1)
    for alpha in np.arange(0.01, 1, 0.01):
        for i in np.arange(0, length, 1):
            row_1 = matrix_1[:,i]
            for cell in row_1:
                print(cell.flat[0])