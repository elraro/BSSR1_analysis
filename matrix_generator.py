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


def matrix_generator_face_x_face(file, folder, normalize, name):
    # tree_users = e_t.parse(file)
    # root_users = tree_users.getroot()
    # length = int(len(root_users)/2)
    # matrix_even = np.empty((length, length))
    # matrix_odd = np.empty((length, length))
    # count = 0
    # even = True
    # for child_user in root_users:
    #     with open(folder + child_user.attrib["name"]) as f:
    #         lines = f.readlines()
    #         scores = lines[2:]  # remove first 2 elements
    #         scores.pop()  # remove last element
    #         scores = [float(score.strip('\n')) for score in scores]
    #         np.asarray(scores)
    #         if even:
    #             matrix_even[count] = scores
    #             even = False
    #         else:
    #             matrix_odd[count] = scores
    #             count += 1
    #             even = True
    # if normalize:
    #     matrix_even = normalize_matrix(matrix_even)
    #     matrix_odd = normalize_matrix(matrix_odd)
    # r_matrix = dict()
    # r_matrix["matrix_1"] = np.matrix(matrix_even)
    # r_matrix["matrix_2"] = np.matrix(matrix_odd)
    # r_matrix["aux"] = name
    # return r_matrix

    # TODO hay que reacerlo
    return 0


def normalize_matrix(matrix):
    # Normalize columns.
    mins = np.min(matrix[:, :], axis=0)
    maxs = np.max(matrix[:, :], axis=0)
    normalized_matrix = (matrix[:, :] - mins) / (maxs - mins)
    return np.nan_to_num(normalized_matrix)
