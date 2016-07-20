import matplotlib.pyplot as plt
import bisect

def normalize_list(lst):
    r = max(lst) - min(lst)
    # Normalize
    normal = map(lambda x: (x - min(lst)) / r, lst)
    return list(normal)

def printROC(FNRate, FPRate, name):
    x = [0, 1]
    FNRate = normalize_list(FNRate)
    FPRate = normalize_list(FPRate)

    # FPRate, FNRate = (list(x) for x in zip(*sorted(zip(FPRate, FNRate), reverse=True, key=lambda pair: pair[0])))

    plt.figure()
    plt.plot(x, x, linestyle='dashed', color='red', linewidth=2, label='y = x')
    plt.plot(FNRate, FPRate, linewidth=1, color='blue', alpha=0.5)

    dif = 1
    index = 0
    for y in range(0, len(FPRate)):
        print(str(FPRate[y] - FNRate[y]))
        if FPRate[y] - FNRate[y] < dif:
            if (FPRate[y] - FNRate[y]) > 0:
                dif = FPRate[y] - FNRate[y]
                index = y

    plt.axvline(x=(FNRate[index] + FNRate[index+1])/2, ymin=0, ymax=(FPRate[index] + FPRate[index+1])/2, color='black', linewidth=2, label='EER = ' + str(dif))
    plt.xlabel('False Negative Rate')
    plt.ylabel('False Positive Rate')
    plt.title(name)
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig('/home/alberto/Desktop/test/' + name + '.png')
    plt.close()

def printROC2(FPRate, TPRate, name):
    x = [0, 1]
    FPRate = normalize_list(FPRate)
    TPRate = normalize_list(TPRate)

    plt.figure()
    plt.plot(x, x, linestyle='dashed', color='red', linewidth=2, label='y = x')
    plt.plot(FPRate, TPRate, linewidth=1, color='blue', alpha=0.5)

    dif = 1
    index = 0
    for y in range(0, len(TPRate)):
        print(str(TPRate[y] - FPRate[y]))
        if TPRate[y] - FPRate[y] < dif:
            if (TPRate[y] - FPRate[y]) > 0:
                dif = TPRate[y] - FPRate[y]
                index = y

    plt.axvline(x=(FPRate[index] + FPRate[index+1])/2, ymin=(TPRate[index] + TPRate[index+1])/2, ymax=1, color='black', linewidth=2, label='EER = ' + str(dif))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(name)
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig('/home/alberto/Desktop/test/' + name + '.png')
    plt.close()