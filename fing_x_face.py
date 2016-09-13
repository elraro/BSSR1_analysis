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

# Fusion biometrica
matrix_fusion_list = mf.matrix_fusion(matrix_face_c, matrix_li_v)

# Dibujar curva ROC
matrix_list = list()
matrix_list.append(matrix_face_c)
matrix_list.append(matrix_face_g)
matrix_list.append(matrix_li_v)
matrix_list.append(matrix_ri_v)

pool = Pool(4)
err_values = pool.map(rc.calculate_roc, matrix_fusion_list)
pool.close()
pool.join()

pool = Pool(4)
roc_values = pool.map(rc.calculate_roc, matrix_list)
pool.close()
pool.join()

rc.draw_roc_eer(err_values, "fing_x_face")
rc.draw_roc(roc_values, "fing_x_face")
