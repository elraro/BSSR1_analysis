import matrix_generator as mg
import matrix_fusion as mf
import roc_calculator as rc
from multiprocessing import Pool
from datetime import datetime


startTime = datetime.now()

matrix_face_c = mg.matrix_generator("/home/alberto/Desktop/bssr1/fing_x_face/sets/dos/users.xml",
                                    "/home/alberto/Desktop/bssr1/fing_x_face/sims/dos/face/C/", True, "face c")
matrix_face_g = mg.matrix_generator("/home/alberto/Desktop/bssr1/fing_x_face/sets/dos/users.xml",
                                    "/home/alberto/Desktop/bssr1/fing_x_face/sims/dos/face/G/", True, "face g")
matrix_li_v = mg.matrix_generator("/home/alberto/Desktop/bssr1/fing_x_face/sets/dos/users.xml",
                                  "/home/alberto/Desktop/bssr1/fing_x_face/sims/dos/li/V/", True, "li v")
matrix_ri_v = mg.matrix_generator("/home/alberto/Desktop/bssr1/fing_x_face/sets/dos/users.xml",
                                  "/home/alberto/Desktop/bssr1/fing_x_face/sims/dos/ri/V/", True, "ri v")

# Fusion biometrica
matrix_fusion_list_face_c_face_g = mf.matrix_fusion(matrix_face_c, matrix_face_g)
matrix_fusion_list_face_c_li_v = mf.matrix_fusion(matrix_face_c, matrix_li_v)
matrix_fusion_list_face_c_ri_v = mf.matrix_fusion(matrix_face_c, matrix_ri_v)
matrix_fusion_list_face_g_li_v = mf.matrix_fusion(matrix_face_g, matrix_li_v)
matrix_fusion_list_face_g_ri_v = mf.matrix_fusion(matrix_face_g, matrix_ri_v)
matrix_fusion_list_li_v_ri_v = mf.matrix_fusion(matrix_ri_v, matrix_ri_v)

# Dibujar curva ROC
matrix_list = list()
matrix_list.append(matrix_face_c)
matrix_list.append(matrix_face_g)
matrix_list.append(matrix_li_v)
matrix_list.append(matrix_ri_v)

pool = Pool(20)
eer_values_face_c_face_g = pool.map(rc.calculate_roc, matrix_fusion_list_face_c_face_g)
eer_values_face_c_li_v = pool.map(rc.calculate_roc, matrix_fusion_list_face_c_li_v)
eer_values_face_c_ri_v = pool.map(rc.calculate_roc, matrix_fusion_list_face_c_ri_v)
eer_values_face_g_li_v = pool.map(rc.calculate_roc, matrix_fusion_list_face_g_li_v)
eer_values_face_g_ri_v = pool.map(rc.calculate_roc, matrix_fusion_list_face_g_ri_v)
eer_values_li_v_ri_v = pool.map(rc.calculate_roc, matrix_fusion_list_li_v_ri_v)
roc_values = pool.map(rc.calculate_roc, matrix_list)
pool.close()
pool.join()

rc.draw_roc_eer(eer_values_face_c_face_g, "face c, face g", roc_values[0]["eer"], roc_values[1]["eer"])
rc.draw_roc_eer(eer_values_face_c_li_v, "face c, li v", roc_values[0]["eer"], roc_values[2]["eer"])
rc.draw_roc_eer(eer_values_face_c_ri_v, "face c, ri v", roc_values[0]["eer"], roc_values[3]["eer"])
rc.draw_roc_eer(eer_values_face_g_li_v, "face g, li v", roc_values[1]["eer"], roc_values[2]["eer"])
rc.draw_roc_eer(eer_values_face_g_ri_v, "face g, ri v", roc_values[1]["eer"], roc_values[3]["eer"])
rc.draw_roc_eer(eer_values_li_v_ri_v, "li v, ri v", roc_values[2]["eer"], roc_values[3]["eer"])
rc.draw_roc(roc_values, "fing_x_face")

print("Executed in: " + str(datetime.now() - startTime))
