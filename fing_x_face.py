import matrix_generator as mg
import matrix_fusion as mf
import roc_calculator as rc
from multiprocessing import Pool


matrix_face_c = mg.matrix_generator("/home/alberto/Desktop/bssr1/fing_x_face/sets/dos/users.xml",
                                    "/home/alberto/Desktop/bssr1/fing_x_face/sims/dos/face/C/", True, "face c")
matrix_face_g = mg.matrix_generator("/home/alberto/Desktop/bssr1/fing_x_face/sets/dos/users.xml",
                                    "/home/alberto/Desktop/bssr1/fing_x_face/sims/dos/face/G/", True, "face g")
matrix_li_v = mg.matrix_generator("/home/alberto/Desktop/bssr1/fing_x_face/sets/dos/users.xml",
                                  "/home/alberto/Desktop/bssr1/fing_x_face/sims/dos/li/V/", True, "li v")
matrix_ri_v = mg.matrix_generator("/home/alberto/Desktop/bssr1/fing_x_face/sets/dos/users.xml",
                                  "/home/alberto/Desktop/bssr1/fing_x_face/sims/dos/ri/V/", True, "ri v")

matrix_fusion_list = mf.matrix_fusion(matrix_face_c, matrix_li_v)

pool = Pool(4)
roc_values = pool.map(rc.calculate_roc, matrix_fusion_list)
pool.close()
pool.join()


# Dibujar gráficas ROC

# matrix = list()
# matrix.append(matrix_face_c)
# matrix.append(matrix_face_g)
# matrix.append(matrix_li_v)
# matrix.append(matrix_ri_v)
#
# pool = Pool(4)
# roc_values = pool.map(rc.calculate_roc, matrix)
# pool.close()
# pool.join()
#
# rc.draw_roc(roc_values, "fing_x_face")
