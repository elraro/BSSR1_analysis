import xml.etree.ElementTree as ET
import PrintROC
import numpy as np
from random import randint

treeEnrollees = ET.parse('/home/alberto/Desktop/bssr1/fing_x_face/sets/dos/enrollees.xml')
treeUsers = ET.parse('/home/alberto/Desktop/bssr1/fing_x_face/sets/dos/users.xml')
rootEnrollees = treeEnrollees.getroot()
rootUsers = treeUsers.getroot()

groundwith = -1

umbral = 0.4

FPRate = []
FNRate = []

for childUser in rootUsers:

    groundwith += 1

    with open('/home/alberto/Desktop/bssr1/fing_x_face/sims/dos/face/C/' + childUser.attrib['name']) as f:
        # print(childUser.attrib['name'])
        lines = f.readlines()
        scores = lines[2:] # remove first 2 elements
        scores.pop() # remove last element
        scores = [float(score.strip('\n')) for score in scores]

        FPRate.clear()
        FNRate.clear()

        # una vez los tengo leidos, tengo que calcular cual es TP, TN, FP, FN
        # como? comparando con mi lista de genuine y sabiendo si es genuino de verdad o no (por la posición de
        # la diagonal, tendré que llevar un contador)
        # el umbral que vaya variando

        for inc in np.arange(0,0.4,0.01):

            TP = 0
            TN = 0
            FP = 0
            FN = 0

            # print(umbral + inc)

            for x in range(0, len(scores)):
                if x == groundwith: # Esta es la comparación de su propio score
                    if scores[x] > (umbral + inc):
                        TP += 1
                    else:
                        FN += 1
                else:
                    if scores[x] > (umbral + inc):
                        FP += 1
                    else:
                        TN += 1

            # Voy a calcular ahora los rates

            #TPRateTemp = TP / (TP + FN)
            #TNRateTemp = TN / (TN + FP)
            FPRateTemp = FP / (FP + TN)
            FNRateTemp = FN / (FN + TP)

            # print(TP)
            # print(TN)
            # print(FP)
            # print(FN)
            #
            # print(FPRateTemp)
            # print(FNRateTemp)
            #
            # print("---------------------")

            FPRate.append(FPRateTemp)
            FNRate.append(FNRateTemp)


        PrintROC.printROC(FPRate, FNRate, childUser.attrib['name'].replace("output", "").replace("/", ""))
        print(groundwith)