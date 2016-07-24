import matrix_generator as mg
import roc_calculator as rc

matrix = mg.matrix_generator("/home/alberto/Desktop/bssr1/fing_x_face/sets/dos/users.xml")

roc = rc.calculate_roc(matrix)

