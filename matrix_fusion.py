def matrix_fusion(matrix_1_dict, matrix_2_dict, alpha):
    matrix_1 = matrix_1_dict["matrix"]
    matrix_2 = matrix_2_dict["matrix"]
    matrix_scores = matrix_1 * alpha + (1 - alpha) * matrix_2
    return matrix_scores