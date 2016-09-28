import matrix_generator as mg
import matrix_fusion as mf
import roc_calculator as rc
from multiprocessing import Pool
from datetime import datetime


def find_eer(eer_list, alpha):
    aux = None
    for eer in eer_list:
        if eer["aux"] == alpha:
            aux = eer
            break
    return aux

startTime = datetime.now()

# Voy a leer las matrices
matrix_face_c = mg.matrix_generator("/home/alberto/Desktop/face c.txt", True, "face c")
matrix_face_g = mg.matrix_generator("/home/alberto/Desktop/face g.txt", True, "face g")
matrix_li_v = mg.matrix_generator("/home/alberto/Desktop/li v.txt", True, "li v")
matrix_ri_v = mg.matrix_generator("/home/alberto/Desktop/ri v.txt", True, "ri v")

# Voy a realizar la fusión de las matrices, así ya las tengo previamente calculadas
matrix_fusion_list_face_c_li_v = mf.matrix_fusion(matrix_face_c, matrix_li_v)
matrix_fusion_list_face_c_ri_v = mf.matrix_fusion(matrix_face_c, matrix_ri_v)
matrix_fusion_list_face_g_li_v = mf.matrix_fusion(matrix_face_g, matrix_li_v)
matrix_fusion_list_face_g_ri_v = mf.matrix_fusion(matrix_face_g, matrix_ri_v)

# Me guardo las matrices originales, para luego dibujar la grafica ROC
matrix_list = list()
matrix_list.append(matrix_face_c)
matrix_list.append(matrix_face_g)
matrix_list.append(matrix_li_v)
matrix_list.append(matrix_ri_v)

eer_values = dict()
pool = Pool(20)
eer_values["eer_values_face_c_li_v_train"] = pool.map(rc.calculate_eer_train,
                                                      matrix_fusion_list_face_c_li_v["matrix_train"])
eer_values["eer_values_face_c_ri_v_train"] = pool.map(rc.calculate_eer_train,
                                                      matrix_fusion_list_face_c_ri_v["matrix_train"])
eer_values["eer_values_face_g_li_v_train"] = pool.map(rc.calculate_eer_train,
                                                      matrix_fusion_list_face_g_li_v["matrix_train"])
eer_values["eer_values_face_g_ri_v_train"] = pool.map(rc.calculate_eer_train,
                                                      matrix_fusion_list_face_g_ri_v["matrix_train"])
eer_values["eer_values_face_c_li_v_test"] = pool.map(rc.calculate_eer_train,
                                                     matrix_fusion_list_face_c_li_v["matrix_test"])
eer_values["eer_values_face_c_ri_v_test"] = pool.map(rc.calculate_eer_train,
                                                     matrix_fusion_list_face_c_ri_v["matrix_test"])
eer_values["eer_values_face_g_li_v_test"] = pool.map(rc.calculate_eer_train,
                                                     matrix_fusion_list_face_g_li_v["matrix_test"])
eer_values["eer_values_face_g_ri_v_test"] = pool.map(rc.calculate_eer_train,
                                                     matrix_fusion_list_face_g_ri_v["matrix_test"])
eer_values["eer_values"] = pool.map(rc.calculate_eer_train, matrix_list)
pool.close()
pool.join()

alphas = dict()
alphas["alpha_face_c_li_v_train"] = rc.calculate_alpha(eer_values["eer_values_face_c_li_v_train"])
alphas["alpha_face_c_ri_v_train"] = rc.calculate_alpha(eer_values["eer_values_face_c_ri_v_train"])
alphas["alpha_face_g_li_v_train"] = rc.calculate_alpha(eer_values["eer_values_face_g_li_v_train"])
alphas["alpha_face_g_ri_v_train"] = rc.calculate_alpha(eer_values["eer_values_face_g_ri_v_train"])

rc.draw_compare_eer(eer_values["eer_values_face_c_li_v_train"], "Fusion face C - finger li",
                    eer_values["eer_values"][0]["eer"], eer_values["eer_values"][2]["eer"], "face c", "finger li")
rc.draw_compare_eer(eer_values["eer_values_face_c_ri_v_train"], "Fusion face C - finger ri",
                    eer_values["eer_values"][0]["eer"], eer_values["eer_values"][3]["eer"], "face c", "finger ri")
rc.draw_compare_eer(eer_values["eer_values_face_g_li_v_train"], "Fusion face g - finger li",
                    eer_values["eer_values"][1]["eer"], eer_values["eer_values"][2]["eer"], "face g", "finger li")
rc.draw_compare_eer(eer_values["eer_values_face_g_ri_v_train"], "Fusion face g - finger ri",
                    eer_values["eer_values"][1]["eer"], eer_values["eer_values"][3]["eer"], "face g", "finger ri")

rc.draw_roc(eer_values["eer_values"], "Finger x Face")

compare = list()
compare.append(eer_values["eer_values"][0])
compare.append(eer_values["eer_values"][2])
compare.append(find_eer(eer_values["eer_values_face_c_li_v_test"], alphas["alpha_face_c_li_v_train"]))
rc.draw_roc(compare, "Fusion face C - finger li EER", "fusion")

compare = list()
compare.append(eer_values["eer_values"][0])
compare.append(eer_values["eer_values"][3])
compare.append(find_eer(eer_values["eer_values_face_c_ri_v_test"], alphas["alpha_face_c_ri_v_train"]))
rc.draw_roc(compare, "Fusion face C - finger ri EER", "fusion")

compare = list()
compare.append(eer_values["eer_values"][1])
compare.append(eer_values["eer_values"][2])
compare.append(find_eer(eer_values["eer_values_face_g_li_v_test"], alphas["alpha_face_g_li_v_train"]))
rc.draw_roc(compare, "Fusion face G - finger li EER", "fusion")

compare = list()
compare.append(eer_values["eer_values"][1])
compare.append(eer_values["eer_values"][3])
compare.append(find_eer(eer_values["eer_values_face_g_ri_v_test"], alphas["alpha_face_g_ri_v_train"]))
rc.draw_roc(compare, "Fusion face G - finger ri EER", "fusion")

print("Executed in: " + str(datetime.now() - startTime))




