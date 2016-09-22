import xml.etree.ElementTree as eT
import numpy as np


def txt_generator(file, folder, name):
    tree_users = eT.parse(file)
    root_users = tree_users.getroot()
    length = len(root_users)
    matrix = np.empty([length, length])
    count = 0
    for child_user in root_users:
        with open(folder + child_user.attrib["name"]) as f:
            lines = f.readlines()
            scores = lines[2:]  # remove first 2 elements
            scores.pop()  # remove last element
            scores = [float(score.strip('\n').strip('\r\n')) for score in scores]
            matrix[count] = scores
            count += 1
    np.savetxt("/home/alberto/Desktop/" + name + ".txt", matrix, fmt="%.7e", delimiter=",", newline="\n",
               header="Matrix " + name, footer="End Matrix", comments='# ')


txt_generator("/home/alberto/Desktop/bssr1/fing_x_face/sets/dos/users.xml",
              "/home/alberto/Desktop/bssr1/fing_x_face/sims/dos/face/C/", "face c")
txt_generator("/home/alberto/Desktop/bssr1/fing_x_face/sets/dos/users.xml",
              "/home/alberto/Desktop/bssr1/fing_x_face/sims/dos/face/G/", "face g")
txt_generator("/home/alberto/Desktop/bssr1/fing_x_face/sets/dos/users.xml",
              "/home/alberto/Desktop/bssr1/fing_x_face/sims/dos/li/V/", "li v")
txt_generator("/home/alberto/Desktop/bssr1/fing_x_face/sets/dos/users.xml",
              "/home/alberto/Desktop/bssr1/fing_x_face/sims/dos/ri/V/", "ri v")
