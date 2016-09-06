import matrix_generator as mg
import roc_calculator as rc
from multiprocessing import Pool

matrix = list()

matrix.append(mg.matrix_generator("/home/alberto/Desktop/bssr1/fing_x_fing/sets/dos/li/users.xml",
                                  "/home/alberto/Desktop/bssr1/fing_x_fing/sims/dos/li/V/", True, "li v"))
matrix.append(mg.matrix_generator("/home/alberto/Desktop/bssr1/fing_x_fing/sets/dos/ri/users.xml",
                                  "/home/alberto/Desktop/bssr1/fing_x_fing/sims/dos/ri/V/", True, "ri v"))

pool = Pool(2)
roc_values = pool.map(rc.calculate_roc, matrix)
pool.close()
pool.join()

rc.draw_roc(roc_values, "fing_x_fing")
