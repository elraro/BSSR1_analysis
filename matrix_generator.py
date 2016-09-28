import numpy as np


def partition_matrix(matrix, percent_test):
    # Train and test partitions
    total = len(matrix)
    train = int(total * percent_test)
    matrix_train = matrix[0:train, 0:train]
    matrix_test = matrix[train:total, train:total]
    return matrix_train, matrix_test


def matrix_generator(file, normalize, name):
    matrix = np.genfromtxt(file, dtype=np.float, comments="#", delimiter=",")
    if normalize:
        matrix = normalize_matrix(matrix)
    r_matrix = dict()
    r_matrix["matrix"] = matrix
    matrix_train, matrix_test = partition_matrix(r_matrix["matrix"], 0.33)
    r_matrix["aux"] = name
    r_matrix["matrix_train"] = matrix_train
    r_matrix["matrix_test"] = matrix_test
    return r_matrix


def normalize_matrix(matrix):
    # Normalize columns.
    mins = np.min(matrix[:, :], axis=0)
    maxs = np.max(matrix[:, :], axis=0)
    normalized_matrix = (matrix[:, :] - mins) / (maxs - mins)
    return np.nan_to_num(normalized_matrix)
