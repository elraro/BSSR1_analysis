import matplotlib.pyplot as plt
import bisect

def normalize_list(lst):
    r = max(lst) - min(lst)
    # Normalize
    normal = map(lambda x: (x - min(lst)) / r, lst)
    return list(normal)

def printROC(FPRate, FNRate, name):
    x = [0, 1]
    FPRate = normalize_list(FPRate)
    FNRate = normalize_list(FNRate)

    FPRate, FNRate = (list(x) for x in zip(*sorted(zip(FPRate, FNRate), reverse=True, key=lambda pair: pair[0])))

    plt.figure()
    plt.plot(x, x, linestyle='dashed', color='red', linewidth=2, label='y = x')
    plt.plot(FNRate, FPRate, linewidth=3, color='blue', alpha=0.5)

    dif = 1
    index = 0
    for y in range(0, len(FNRate)):
        if (FPRate[y] - FNRate[y]) < dif:
            if (FPRate[y] - FNRate[y]) > 0:
                dif = FPRate[y] - FNRate[y]
                index = y

    print(dif)
    print(index)
    print(FNRate[index])
    print(FPRate[index])

    plt.axvline(x=(FNRate[index] + FNRate[index+1])/2, ymin=0, ymax=(FPRate[index] + FPRate[index+1])/2, color='black', linewidth=2, label='EER = ' + str((FNRate[index] + FNRate[index+1])/2))
    plt.xlabel('% FN')
    plt.ylabel('% FP')
    plt.title(name)
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig('/home/alberto/Desktop/test/' + name + '.png')
    plt.close()