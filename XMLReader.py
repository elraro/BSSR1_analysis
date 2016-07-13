import xml.etree.ElementTree as ET
import PrintROC

treeEnrollees = ET.parse('/home/alberto/Desktop/bssr1/fing_x_face/sets/dos/enrollees.xml')
treeUsers = ET.parse('/home/alberto/Desktop/bssr1/fing_x_face/sets/dos/users.xml')
rootEnrollees = treeEnrollees.getroot()
rootUsers = treeUsers.getroot()

genuine = []
matrix = []

groundwith = -1

umbral = 0.538

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
        genuineAux = []
        matrix.append(scores)
        for score in scores:
            if score > umbral: # esto tendre que irlo variando para encontrar el mejor umbral
                genuineAux.append(1)
            else:
                genuineAux.append(0)
        genuine.append(genuineAux)

        # una vez los tengo generados con mi umbral, tengo que calcular cual es TP, TN, FP, FN
        # como? comparando con mi lista de genuine y sabiendo si es genuino de verdad o no (por la posición de
        # la diagonal, tendré que llevar un contador

        TP = 0
        TN = 0
        FP = 0
        FN = 0

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

        #TPRateTemp = TP / (TP + FN)
        #TNRateTemp = TN / (TN + FP)
        FPRateTemp = FP / (FP + TN)
        FNRateTemp = FN / (FN + TP)

        print(TP)
        print(TN)
        print(FP)
        print(FN)

        print(FPRateTemp)
        print(FNRateTemp)

        print("---------------------")

        FPRate.append(FPRateTemp)
        FNRate.append(FNRateTemp)

trueLabel = []


# Rough code
# Get the range of the list
r = max(scores) - min(scores)
# Normalize
normal = map(lambda x: (x - min(scores)) / r, scores)
# print(list(normal))
scores = list(normal)

for score in scores:
    if score >= 0.5:
        trueLabel.append(1)
    else:
        trueLabel.append(0)
# print(trueLabel)

PrintROC.printROC(FPRate, FNRate)