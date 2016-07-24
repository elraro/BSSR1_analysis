import numpy as np
import matplotlib.pyplot as plt

def normalize_list(lst):
    r = max(lst) - min(lst)
    # Normalize
    normal = map(lambda x: (x - min(lst)) / r, lst)
    return list(normal)

def calculate_roc(matrix):
    identity = np.identity(517)
    TPRate = np.empty(shape=0)
    TNRate = np.empty(shape=0)
    FNRate = np.empty(shape=0)
    FPRate = np.empty(shape=0)
    for umbral in np.arange(0, 1, 0.01):
        TP = np.logical_and(umbral > matrix, identity == 1).sum()
        TN = np.logical_and(umbral <= matrix, identity == 0).sum()
        FP = np.logical_and(umbral > matrix, identity == 0).sum()
        FN = np.logical_and(umbral <= matrix, identity == 1).sum()
        TPRate = np.append(TPRate, TP / (TP + FN))
        TNRate = np.append(TNRate, TN / (TN + FP))
        FNRate = np.append(FNRate, FN / (FN + + TP))
        FPRate = np.append(FPRate, FP / (FP + TN))
    # print(TP)
    # print(TN)
    # print(FP)
    # print(FN)

    print(FNRate)
    print(FPRate)
    plt.figure()
    plt.plot(FNRate[::-1], FPRate[::-1], linewidth=1, color='blue', alpha=0.5)
    # dif = 1
    # index = 0
    # for y in range(0, len(FPRate)):
    #     print(str(abs(FPRate[y] - FNRate[y])))
    # print(index)
    # if abs(FPRate[y] - FNRate[y]) < dif:
    #     dif = abs(FPRate[y] - FNRate[y])
    # index = y
    # print("eer: " + str((FPRate[index] + FNRate[index]) / 2))
    # print("index: " + str(index))
    # plt.axvline(x=(FNRate[index] + FNRate[index + 1]) / 2, ymin=0, ymax=(FPRate[index] + FPRate[index + 1]) / 2,
    #             color='black', linewidth=2, label='EER = ' + str((FNRate[index] + FNRate[index + 1]) / 2))
    plt.xlabel('False Negative Rate')
    plt.ylabel('False Positive Rate')
    plt.title("roc")
    plt.legend(loc="lower right")
    plt.show()
    # plt.savefig('/home/alberto/Desktop/test/' + name + '.png')
    plt.close()