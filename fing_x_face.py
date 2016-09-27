import matrix_generator as mg
import matrix_fusion as mf
import roc_calculator as rc
from multiprocessing import Pool
from datetime import datetime

startTime = datetime.now()

matrix_face_c = mg.matrix_generator("/home/alberto/Desktop/face c.txt", True, "face c")
matrix_face_g = mg.matrix_generator("/home/alberto/Desktop/face g.txt", True, "face g")
matrix_li_v = mg.matrix_generator("/home/alberto/Desktop/li v.txt", True, "li v")
matrix_ri_v = mg.matrix_generator("/home/alberto/Desktop/ri v.txt", True, "ri v")

# Fusion biometrica
matrix_fusion_list_face_c_face_g = mf.matrix_fusion(matrix_face_c, matrix_face_g)
matrix_fusion_list_face_c_li_v = mf.matrix_fusion(matrix_face_c, matrix_li_v)
matrix_fusion_list_face_c_ri_v = mf.matrix_fusion(matrix_face_c, matrix_ri_v)
matrix_fusion_list_face_g_face_c = mf.matrix_fusion(matrix_face_g, matrix_face_c)
matrix_fusion_list_face_g_li_v = mf.matrix_fusion(matrix_face_g, matrix_li_v)
matrix_fusion_list_face_g_ri_v = mf.matrix_fusion(matrix_face_g, matrix_ri_v)
matrix_fusion_list_li_v_ri_v = mf.matrix_fusion(matrix_ri_v, matrix_ri_v)

# Dibujar curva ROC
matrix_list = list()
matrix_list.append(matrix_face_c)
matrix_list.append(matrix_face_g)
matrix_list.append(matrix_li_v)
matrix_list.append(matrix_ri_v)

eer_values = dict()
pool = Pool(20)
eer_values["eer_values_face_c_face_g"] = pool.map(rc.calculate_roc, matrix_fusion_list_face_c_face_g["matrix"])
eer_values["eer_values_face_c_li_v"] = pool.map(rc.calculate_roc, matrix_fusion_list_face_c_li_v["matrix"])
eer_values["eer_values_face_c_ri_v"] = pool.map(rc.calculate_roc, matrix_fusion_list_face_c_ri_v["matrix"])
eer_values["eer_values_face_g_face_c"] = pool.map(rc.calculate_roc, matrix_fusion_list_face_g_face_c["matrix"])
eer_values["eer_values_face_g_li_v"] = pool.map(rc.calculate_roc, matrix_fusion_list_face_g_li_v["matrix"])
eer_values["eer_values_face_g_ri_v"] = pool.map(rc.calculate_roc, matrix_fusion_list_face_g_ri_v["matrix"])
eer_values["eer_values_li_v_ri_v"] = pool.map(rc.calculate_roc, matrix_fusion_list_li_v_ri_v["matrix"])
eer_values["eer_values_face_c_face_g_train"] = pool.map(rc.calculate_roc,
                                                        matrix_fusion_list_face_c_face_g["matrix_train"])
eer_values["eer_values_face_c_li_v_train"] = pool.map(rc.calculate_roc, matrix_fusion_list_face_c_li_v["matrix_train"])
eer_values["eer_values_face_c_ri_v_train"] = pool.map(rc.calculate_roc, matrix_fusion_list_face_c_ri_v["matrix_train"])
eer_values["eer_values_face_g_face_c_train"] = pool.map(rc.calculate_roc,
                                                        matrix_fusion_list_face_g_face_c["matrix_train"])
eer_values["eer_values_face_g_li_v_train"] = pool.map(rc.calculate_roc, matrix_fusion_list_face_g_li_v["matrix_train"])
eer_values["eer_values_face_g_ri_v_train"] = pool.map(rc.calculate_roc, matrix_fusion_list_face_g_ri_v["matrix_train"])
eer_values["eer_values_li_v_ri_v_train"] = pool.map(rc.calculate_roc, matrix_fusion_list_li_v_ri_v["matrix_train"])
eer_values["eer_values_face_c_face_g_test"] = pool.map(rc.calculate_roc,
                                                       matrix_fusion_list_face_c_face_g["matrix_test"])
eer_values["eer_values_face_c_li_v_test"] = pool.map(rc.calculate_roc, matrix_fusion_list_face_c_li_v["matrix_test"])
eer_values["eer_values_face_c_ri_v_test"] = pool.map(rc.calculate_roc, matrix_fusion_list_face_c_ri_v["matrix_test"])
eer_values["eer_values_face_g_face_c_test"] = pool.map(rc.calculate_roc,
                                                       matrix_fusion_list_face_g_face_c["matrix_test"])
eer_values["eer_values_face_g_li_v_test"] = pool.map(rc.calculate_roc, matrix_fusion_list_face_g_li_v["matrix_test"])
eer_values["eer_values_face_g_ri_v_test"] = pool.map(rc.calculate_roc, matrix_fusion_list_face_g_ri_v["matrix_test"])
eer_values["eer_values_li_v_ri_v_test"] = pool.map(rc.calculate_roc, matrix_fusion_list_li_v_ri_v["matrix_test"])
eer_values["roc_values"] = pool.map(rc.calculate_roc, matrix_list)
pool.close()
pool.join()

alphas = dict()
alphas["alpha_face_c_face_g_total"] = rc.draw_roc_eer(eer_values["eer_values_face_c_face_g"], "face c, face g total",
                                                      eer_values["roc_values"][0]["eer"],
                                                      eer_values["roc_values"][1]["eer"])
alphas["alpha_face_c_li_v_total"] = rc.draw_roc_eer(eer_values["eer_values_face_c_li_v"], "face c, li v total",
                                                    eer_values["roc_values"][0]["eer"],
                                                    eer_values["roc_values"][2]["eer"])
alphas["alpha_face_c_ri_v_total"] = rc.draw_roc_eer(eer_values["eer_values_face_c_ri_v"], "face c, ri v total",
                                                    eer_values["roc_values"][0]["eer"],
                                                    eer_values["roc_values"][3]["eer"])
alphas["alpha_face_g_face_c_total"] = rc.draw_roc_eer(eer_values["eer_values_face_g_face_c"], "face g, face c total",
                                                    eer_values["roc_values"][1]["eer"],
                                                    eer_values["roc_values"][0]["eer"])
alphas["alpha_face_g_li_v_total"] = rc.draw_roc_eer(eer_values["eer_values_face_g_li_v"], "face g, li v total",
                                                    eer_values["roc_values"][1]["eer"],
                                                    eer_values["roc_values"][2]["eer"])
alphas["alpha_face_g_ri_v_total"] = rc.draw_roc_eer(eer_values["eer_values_face_g_ri_v"], "face g, ri v total",
                                                    eer_values["roc_values"][1]["eer"],
                                                    eer_values["roc_values"][3]["eer"])
alphas["alpha_li_v_ri_v_total"] = rc.draw_roc_eer(eer_values["eer_values_li_v_ri_v"], "li v, ri v total",
                                                  eer_values["roc_values"][2]["eer"],
                                                  eer_values["roc_values"][3]["eer"])
alphas["alpha_face_c_face_g_train"] = rc.draw_roc_eer(eer_values["eer_values_face_c_face_g_train"],
                                                      "face c, face g train", eer_values["roc_values"][0]["eer"],
                                                      eer_values["roc_values"][1]["eer"])
alphas["alpha_face_c_li_v_train"] = rc.draw_roc_eer(eer_values["eer_values_face_c_li_v_train"], "face c, li v train",
                                                    eer_values["roc_values"][0]["eer"],
                                                    eer_values["roc_values"][2]["eer"])
alphas["alpha_face_c_ri_v_train"] = rc.draw_roc_eer(eer_values["eer_values_face_c_ri_v_train"], "face c, ri v train",
                                                    eer_values["roc_values"][0]["eer"],
                                                    eer_values["roc_values"][3]["eer"])
alphas["alpha_face_g_face_c_train"] = rc.draw_roc_eer(eer_values["eer_values_face_g_face_c_train"], "face g, face c train",
                                                      eer_values["roc_values"][1]["eer"],
                                                      eer_values["roc_values"][0]["eer"])
alphas["alpha_face_g_li_v_train"] = rc.draw_roc_eer(eer_values["eer_values_face_g_li_v_train"], "face g, li v train",
                                                    eer_values["roc_values"][1]["eer"],
                                                    eer_values["roc_values"][2]["eer"])
alphas["alpha_face_g_ri_v_train"] = rc.draw_roc_eer(eer_values["eer_values_face_g_ri_v_train"], "face g, ri v train",
                                                    eer_values["roc_values"][1]["eer"],
                                                    eer_values["roc_values"][3]["eer"])
alphas["alpha_li_v_ri_v_train"] = rc.draw_roc_eer(eer_values["eer_values_li_v_ri_v_train"], "li v, ri v train",
                                                  eer_values["roc_values"][2]["eer"],
                                                  eer_values["roc_values"][3]["eer"])
alphas["alpha_face_c_face_g_test"] = rc.draw_roc_eer(eer_values["eer_values_face_c_face_g_test"], "face c, face g test",
                                                     eer_values["roc_values"][0]["eer"],
                                                     eer_values["roc_values"][1]["eer"])
alphas["alpha_face_c_li_v_test"] = rc.draw_roc_eer(eer_values["eer_values_face_c_li_v_test"], "face c, li v test",
                                                   eer_values["roc_values"][0]["eer"],
                                                   eer_values["roc_values"][2]["eer"])
alphas["alpha_face_c_ri_v_test"] = rc.draw_roc_eer(eer_values["eer_values_face_c_ri_v_test"], "face c, ri v test",
                                                   eer_values["roc_values"][0]["eer"],
                                                   eer_values["roc_values"][3]["eer"])
alphas["alpha_face_g_face_c_test"] = rc.draw_roc_eer(eer_values["eer_values_face_g_face_c_test"], "face g, face c test",
                                                    eer_values["roc_values"][1]["eer"],
                                                    eer_values["roc_values"][0]["eer"])
alphas["alpha_face_g_li_v_test"] = rc.draw_roc_eer(eer_values["eer_values_face_g_li_v_test"], "face g, li v test",
                                                   eer_values["roc_values"][1]["eer"],
                                                   eer_values["roc_values"][2]["eer"])
alphas["alpha_face_g_ri_v_test"] = rc.draw_roc_eer(eer_values["eer_values_face_g_ri_v_test"], "face g, ri v test",
                                                   eer_values["roc_values"][1]["eer"],
                                                   eer_values["roc_values"][3]["eer"])
alphas["alpha_li_v_ri_v_test"] = rc.draw_roc_eer(eer_values["eer_values_li_v_ri_v_test"], "li v, ri v test",
                                                 eer_values["roc_values"][2]["eer"], eer_values["roc_values"][3]["eer"])
rc.draw_roc(eer_values["roc_values"], "fing_x_face")

rc.draw_roc_alphas(alphas["alpha_face_c_face_g_test"], alphas["alpha_face_c_li_v_test"],
                   alphas["alpha_face_c_ri_v_test"], eer_values["eer_values_face_c_face_g"],
                   eer_values["eer_values_face_c_li_v"], eer_values["eer_values_face_c_ri_v"], "face c face g ",
                   "face c li v ", "face c ri v ", "face c alphas")

rc.draw_roc_alphas(alphas["alpha_face_g_face_c_test"], alphas["alpha_face_g_li_v_test"],
                   alphas["alpha_face_g_ri_v_test"], eer_values["eer_values_face_c_face_g"],
                   eer_values["eer_values_face_g_li_v"], eer_values["eer_values_face_g_ri_v"], "face g face c ",
                   "face g li v ", "face g ri v ", "face g alphas")

print("Executed in: " + str(datetime.now() - startTime))
