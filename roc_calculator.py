import matplotlib.pyplot as plt

def normalize_list(lst):
    r = max(lst) - min(lst)
    # Normalize
    normal = map(lambda x: (x - min(lst)) / r, lst)
    return list(normal)

def calculate_roc(matrix, name):
    x = [0, 1]
    # FNRate = normalize_list(FNRate)
    # FPRate = normalize_list(FPRate)

    # FPRate, FNRate = (list(x) for x in zip(*sorted(zip(FPRate, FNRate), reverse=True, key=lambda pair: pair[0])))

    plt.figure()
    plt.plot(x, x, linestyle='dashed', color='red', linewidth=2, label='y = x')
    plt.plot(fn_rate, fp_rate, linewidth=1, color='blue', alpha=0.5)

    dif = 1
    index = 0
    for y in range(0, len(fp_rate)):
        if abs(fp_rate[y] - fn_rate[y]) < dif:
            dif = abs(fp_rate[y] - fn_rate[y])
            index = y

    print("eer: " + str((fp_rate[index] + fn_rate[index])/2))
    print("index: " + str(index))
    plt.axvline(x=(fn_rate[index] + fn_rate[index+1])/2, ymin=0, ymax=(fp_rate[index] + fp_rate[index+1])/2, color='black', linewidth=2, label='EER = ' + str((fn_rate[index] + fn_rate[index+1])/2))
    plt.xlabel('False Negative Rate')
    plt.ylabel('False Positive Rate')
    plt.title(name)
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig('/home/alberto/Desktop/test/' + name + '.png')
    plt.close()