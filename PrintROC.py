import matplotlib.pyplot as plt
import numpy as np

def normalize_list(lst):
    r = max(lst) - min(lst)
    # Normalize
    normal = map(lambda x: (x - min(lst)) / r, lst)
    return list(normal)

def printROC(FNRate, FPRate, name):
    x = [0, 1]
    # FNRate = normalize_list(FNRate)
    # FPRate = normalize_list(FPRate)

    # FPRate, FNRate = (list(x) for x in zip(*sorted(zip(FPRate, FNRate), reverse=True, key=lambda pair: pair[0])))

    plt.figure()
    plt.plot(x, x, linestyle='dashed', color='red', linewidth=2, label='y = x')
    plt.plot(FNRate, FPRate, linewidth=1, color='blue', alpha=0.5)

    dif = 1
    index = 0
    for y in range(0, len(FPRate)):
        print(str(abs(FPRate[y] - FNRate[y])))
        print(index)
        if abs(FPRate[y] - FNRate[y]) < dif:
            dif = abs(FPRate[y] - FNRate[y])
            index = y

    print("eer: " + str((FPRate[index] + FNRate[index])/2))
    print("index: " + str(index))
    plt.axvline(x=(FNRate[index] + FNRate[index+1])/2, ymin=0, ymax=(FPRate[index] + FPRate[index+1])/2, color='black', linewidth=2, label='EER = ' + str((FNRate[index] + FNRate[index+1])/2))
    plt.xlabel('False Negative Rate')
    plt.ylabel('False Positive Rate')
    plt.title(name)
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig('/home/alberto/Desktop/test/' + name + '.png')
    plt.close()