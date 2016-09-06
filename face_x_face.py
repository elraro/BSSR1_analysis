import matrix_generator as mg
import roc_calculator as rc
from multiprocessing import Pool

matrix = list()

matrix_1 = mg.matrix_generator_face_x_face("/home/alberto/Desktop/bssr1/face_x_face/sets/dos/users.xml",
                                           "/home/alberto/Desktop/bssr1/face_x_face/sims/dos/face/C/", False, "face c")
matrix_2 = mg.matrix_generator_face_x_face("/home/alberto/Desktop/bssr1/face_x_face/sets/dos/users.xml",
                                           "/home/alberto/Desktop/bssr1/face_x_face/sims/dos/face/G/", True, "face g")

r_matrix = dict()
r_matrix["matrix"] = matrix_1["matrix_1"]
r_matrix["name"] = "face c 1"
matrix.append(r_matrix)

r_matrix = dict()
r_matrix["matrix"] = matrix_1["matrix_2"]
r_matrix["name"] = "face c 2"
matrix.append(r_matrix)

r_matrix = dict()
r_matrix["matrix"] = matrix_2["matrix_1"]
r_matrix["name"] = "face g 1"
matrix.append(r_matrix)

r_matrix = dict()
r_matrix["matrix"] = matrix_2["matrix_2"]
r_matrix["name"] = "face g 2"
matrix.append(r_matrix)

pool = Pool(4)
roc_values = pool.map(rc.calculate_roc, matrix)
pool.close()
pool.join()

rc.draw_roc(roc_values, "face_x_face")
