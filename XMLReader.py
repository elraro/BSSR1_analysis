import xml.etree.ElementTree as ET
import PrintROC
import numpy as np

treeUsers = ET.parse('/home/alberto/Desktop/bssr1/fing_x_face/sets/dos/users.xml')
#treeUsers = ET.parse('/home/alberto/Desktop/bssr1/face_x_face/sets/dos/users.xml')
rootUsers = treeUsers.getroot()

TPRate = []
TNRate = []
FNRate = []
FPRate = []

TP = 0
TN = 0
FP = 0
FN = 0

for umbral in np.arange(0, 1, 0.01):

    groundwith = -1

    for childUser in rootUsers:

        groundwith += 1

        with open('/home/alberto/Desktop/bssr1/fing_x_face/sims/dos/face/C/' + childUser.attrib['name']) as f:
        #with open("/home/alberto/Desktop/bssr1/face_x_face/sims/dos/face/C/" + childUser.attrib['name']) as f:
            lines = f.readlines()
            scores = lines[2:] # remove first 2 elements
            scores.pop() # remove last element
            scores = [float(score.strip('\n')) for score in scores]

            # una vez los tengo leidos, tengo que calcular cual es TP, TN, FP, FN
            # como? comparando con mi lista de genuine y sabiendo si es genuino de verdad o no (por la posición de
            # la diagonal, tendré que llevar un contador)
            # el umbral que vaya variando

            for x in range(0, len(scores)):
                if x == groundwith: # Esta es la comparación de su propio score
                    if scores[x] > umbral:
                        TP += 1
                    else:
                        FN += 1
                else:
                    if scores[x] > umbral:
                        FP += 1
                    else:
                        TN += 1

    # Voy a calcular ahora los rates
    TPRate.append(TP / (TP + FN))
    TNRate.append(TN / (TN + FP))
    FNRate.append(FN / (FN + + TP))
    FPRate.append(FP / (FP + TN))

PrintROC.printROC(FNRate, FPRate, "Curva_ROC_FN_FP")
# PrintROC.printROC2(FPRate, TPRate, "Curva_ROC_FP_TP")