import xml.etree.ElementTree as e_t
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit


def partition_matrix(matrix):
    # Train and test partitions
    y = matrix[:, -1]
    sss = StratifiedShuffleSplit(y, 1, test_size=0.5, random_state=1)
    normalized_matrix = []
    normalized_matrix2 = []
    for train_index, test_index in sss:
        for i in train_index:
            normalized_matrix.append(matrix[i])
        for i in test_index:
            normalized_matrix2.append(matrix[i])
    normalized_matrix = np.array(normalized_matrix)
    normalized_matrix2 = np.array(normalized_matrix2)
    return normalized_matrix, normalized_matrix2


def matrix_generator(file, folder, normalize, name):
    tree_users = e_t.parse(file)
    root_users = tree_users.getroot()
    length = len(root_users)
    matrix = np.empty((length, length))
    count = 0
    for child_user in root_users:
        with open(folder + child_user.attrib["name"]) as f:
            lines = f.readlines()
            scores = lines[2:]  # remove first 2 elements
            scores.pop()  # remove last element
            scores = [float(score.strip('\n').strip('\r\n')) for score in scores]
            np.asarray(scores)
            matrix[count] = scores
            count += 1
    if normalize:
        matrix = normalize_matrix(matrix)
    r_matrix = dict()
    r_matrix["matrix_total"] = np.matrix(matrix)
    training, test = partition_matrix(r_matrix["matrix_total"])
    r_matrix["aux"] = name
    r_matrix["matrix"] = training
    r_matrix["test"] = test
    return r_matrix


def matrix_generator_face_x_face(file, folder, normalize, name):
    tree_users = e_t.parse(file)
    root_users = tree_users.getroot()
    length = int(len(root_users)/2)
    matrix_even = np.empty((length, length))
    matrix_odd = np.empty((length, length))
    count = 0
    even = True
    for child_user in root_users:
        with open(folder + child_user.attrib["name"]) as f:
            lines = f.readlines()
            scores = lines[2:]  # remove first 2 elements
            scores.pop()  # remove last element
            scores = [float(score.strip('\n')) for score in scores]
            np.asarray(scores)
            if even:
                matrix_even[count] = scores
                even = False
            else:
                matrix_odd[count] = scores
                count += 1
                even = True
    if normalize:
        matrix_even = normalize_matrix(matrix_even)
        matrix_odd = normalize_matrix(matrix_odd)
    r_matrix = dict()
    r_matrix["matrix_1"] = np.matrix(matrix_even)
    r_matrix["matrix_2"] = np.matrix(matrix_odd)
    r_matrix["aux"] = name
    return r_matrix


def normalize_matrix(matrix):
    # Normalize columns.
    mins = np.min(matrix[:, :], axis=0)
    maxs = np.max(matrix[:, :], axis=0)
    normalized_matrix = (matrix[:, :] - mins) / (maxs - mins)
    return np.nan_to_num(normalized_matrix)