import matrix_generator as mg
import matrix_fusion as mf
import roc_calculator as rc
from multiprocessing import Pool

matrix = list()

matrix_face_c = mg.matrix_generator("/home/alberto/Desktop/bssr1/fing_x_face/sets/dos/users.xml",
                                    "/home/alberto/Desktop/bssr1/fing_x_face/sims/dos/face/C/", False, "face c")
matrix_face_g = mg.matrix_generator("/home/alberto/Desktop/bssr1/fing_x_face/sets/dos/users.xml",
                                    "/home/alberto/Desktop/bssr1/fing_x_face/sims/dos/face/G/", True, "face g")
matrix_li_v = mg.matrix_generator("/home/alberto/Desktop/bssr1/fing_x_face/sets/dos/users.xml",
                                  "/home/alberto/Desktop/bssr1/fing_x_face/sims/dos/li/V/", True, "li v")
mattrix_ri_v = mg.matrix_generator("/home/alberto/Desktop/bssr1/fing_x_face/sets/dos/users.xml",
                                   "/home/alberto/Desktop/bssr1/fing_x_face/sims/dos/ri/V/", True, "ri v")

mf.matrix_fusion(matrix_face_c, matrix_li_v)

matrix.append(matrix_face_c)
matrix.append(matrix_face_g)
matrix.append(matrix_li_v)
matrix.append(mattrix_ri_v)

pool = Pool(4)
roc_values = pool.map(rc.calculate_roc, matrix)
pool.close()
pool.join()

rc.draw_roc(roc_values, "fing_x_face")
