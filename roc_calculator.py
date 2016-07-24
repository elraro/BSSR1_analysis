import numpy as np

def normalize_list(lst):
    r = max(lst) - min(lst)
    # Normalize
    normal = map(lambda x: (x - min(lst)) / r, lst)
    return list(normal)

def calculate_roc(matrix):
    identity = np.identity(517)
    tp_rate = np.empty(shape=0)
    tn_rate = np.empty(shape=0)
    fn_rate = np.empty(shape=0)
    fp_rate = np.empty(shape=0)
    for umbral in np.arange(0, 1.01, 0.01):
        # http://notmatthancock.github.io/2015/08/19/roc-curve-part-2-numerical-example.html
        TP = np.logical_and(matrix >= umbral, identity == 1).sum()
        TN = np.logical_and(matrix < umbral, identity == 0).sum()
        FP = np.logical_and(matrix >= umbral, identity == 0).sum()
        FN = np.logical_and(matrix < umbral, identity == 1).sum()
        tp_rate = np.append(tp_rate, TP / (TP + FN))
        tn_rate = np.append(tn_rate, TN / (TN + FP))
        fn_rate = np.append(fn_rate, FN / (FN + + TP))
        fp_rate = np.append(fp_rate, FP / (FP + TN))

    dif = 1
    index = 0
    for y in range(0, len(fp_rate)):
        if abs(fp_rate[y] - fp_rate[y]) < dif:
            dif = abs(fp_rate[y] - fp_rate[y])
            index = y

    roc = dict()
    roc["tp_rate"] = tp_rate
    roc["tn_rate"] = tn_rate
    roc["fn_rate"] = fn_rate
    roc["fp_rate"] = fp_rate
    roc["err"] = (fp_rate[index] + fp_rate[index + 1]) / 2

    return roc
    #
    #
    # # FNRate = FNRate[::-1]
    # # FPRate = FPRate[::-1]
    # plt.figure()
    # x = [0, 1]
    # plt.plot(x, x, linestyle='dashed', color='red', linewidth=2, label='y = x')
    # plt.plot(FNRate, FPRate, linewidth=1, color='blue', alpha=0.5)
    #
    # dif = 1
    # index = 0
    # for y in range(0, len(FPRate)):
    #     print(str(abs(FPRate[y] - FNRate[y])))
    #     print(index)
    #     if abs(FPRate[y] - FNRate[y]) < dif:
    #         dif = abs(FPRate[y] - FNRate[y])
    #         index = y
    # print("eer: " + str((FPRate[index] + FNRate[index+1]) / 2))
    # print("index: " + str(index))
    # plt.axvline(x=(FNRate[index] + FNRate[index + 1]) / 2, ymin=0, ymax=(FPRate[index] + FPRate[index + 1]) / 2,
    #              color='black', linewidth=2, label='EER = ' + str((FNRate[index] + FNRate[index + 1]) / 2))
    # plt.xlabel('False Negative Rate')
    # plt.ylabel('False Positive Rate')
    # plt.title("roc")
    # plt.legend(loc="lower right")
    # plt.show()
    # # plt.savefig('/home/alberto/Desktop/test/' + name + '.png')
    # plt.close()