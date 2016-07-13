import matplotlib.pyplot as plt
import numpy as np

def normalize_list(lst):
    r = max(lst) - min(lst)
    # Normalize
    normal = map(lambda x: (x - min(lst)) / r, lst)
    # print(list(normal))
    return list(normal)

def printROC(FPRate, FNRate):
    x = [-0.1, 1.1]
    FPRate = normalize_list(FPRate)
    FNRate = normalize_list(FNRate)

    # FPRate, FNRate = (list(x) for x in zip(*sorted(zip(FPRate, FNRate), reverse=True, key=lambda pair: pair[0])))
    FNRate, FPRate = (list(x) for x in zip(*sorted(zip(FNRate, FPRate), reverse=True, key=lambda pair: pair[0])))

    print(FPRate)
    print(FNRate)

    plt.plot(x, x, linestyle='dashed', color='red', linewidth=2, label='y = x')
    plt.plot(FNRate, FPRate, linewidth=3,color='blue', alpha=0.5)
    plt.xlabel('% FN')
    plt.ylabel('% FP')
    plt.title('ROC example')
    plt.legend(loc="lower right")
    plt.show()