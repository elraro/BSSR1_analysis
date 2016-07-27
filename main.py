import matrix_generator as mg
import roc_calculator as rc

matrix_face_c = mg.matrix_generator("/home/alberto/Desktop/bssr1/fing_x_face/sets/dos/users.xml",
                                    "/home/alberto/Desktop/bssr1/fing_x_face/sims/dos/face/C/", False)
matrix_face_g = mg.matrix_generator("/home/alberto/Desktop/bssr1/fing_x_face/sets/dos/users.xml",
                                    "/home/alberto/Desktop/bssr1/fing_x_face/sims/dos/face/G/", True)
matrix_li_v = mg.matrix_generator("/home/alberto/Desktop/bssr1/fing_x_face/sets/dos/users.xml",
                                    "/home/alberto/Desktop/bssr1/fing_x_face/sims/dos/li/V/", True)
matrix_ri_v = mg.matrix_generator("/home/alberto/Desktop/bssr1/fing_x_face/sets/dos/users.xml",
                                    "/home/alberto/Desktop/bssr1/fing_x_face/sims/dos/ri/V/", True)

roc_face_c = rc.calculate_roc(matrix_face_c, "face c")
roc_face_g = rc.calculate_roc(matrix_face_g, "face g")
roc_li_v = rc.calculate_roc(matrix_li_v, "li v")
roc_ri_v = rc.calculate_roc(matrix_ri_v, "ri v")

rc.draw_roc(roc_face_c, roc_face_g, roc_li_v, roc_ri_v)