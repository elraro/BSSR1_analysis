import xml.etree.ElementTree as e_t
import numpy as np

def matrix_generator(file):
    tree_users = e_t.parse(file)
    root_users = tree_users.getroot()
    length = len(root_users)
    matrix = np.empty((length, length))
    count = 0
    for child_user in root_users:
        with open("/home/alberto/Desktop/bssr1/fing_x_face/sims/dos/face/C/" + child_user.attrib["name"]) as f:
            lines = f.readlines()
            scores = lines[2:]  # remove first 2 elements
            scores.pop()  # remove last element
            scores = [float(score.strip('\n')) for score in scores]
            np.asarray(scores)
            matrix[count] = scores
            count += 1
    return np.matrix(matrix)
