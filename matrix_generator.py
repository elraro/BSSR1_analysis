import xml.etree.ElementTree as ET
import numpy as np

def matrix_generator(file):
    treeUsers = ET.parse(file)
    rootUsers = treeUsers.getroot()

    matrix = np.empty((517, 517)) # hardcode CARE!

    count = 0

    for childUser in rootUsers:
        with open('/home/alberto/Desktop/bssr1/fing_x_face/sims/dos/face/C/' + childUser.attrib['name']) as f:
            lines = f.readlines()
            scores = lines[2:] # remove first 2 elements
            scores.pop() # remove last element
            scores = [float(score.strip('\n')) for score in scores]
            np.asarray(scores)
            matrix[count] = scores
            count += 1
    return np.matrix(matrix)