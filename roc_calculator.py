import numpy as np
import matplotlib.pyplot as plt
from collections import deque


def calculate_roc(matrix_dict):
    matrix = matrix_dict["matrix"]
    aux = matrix_dict["aux"]
    identity = np.identity(len(matrix))
    tp_rate = np.empty(shape=0)
    tn_rate = np.empty(shape=0)
    fn_rate = np.empty(shape=0)
    fp_rate = np.empty(shape=0)
    err_found = False
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
        fn_rate_err = FN / (FN + + TP)
        fp_rate_err = FP / (FP + TN)
        if fn_rate_err > fp_rate_err and not err_found:
            err = (fn_rate_err + fp_rate_err) / 2
            err_found = True
    roc = dict()
    roc["tp_rate"] = tp_rate
    roc["tn_rate"] = tn_rate
    roc["fn_rate"] = fn_rate
    roc["fp_rate"] = fp_rate
    roc["eer"] = err
    roc["aux"] = aux
    return roc


def draw_roc(roc_values, title):
    colours = create_colours()
    plt.figure()
    x = [0, 1]
    plt.plot(x, x, linestyle="dashed", color="red", linewidth=1)
    for roc in roc_values:
        plt.plot(roc["fn_rate"], roc["fp_rate"], linewidth=1, color=colours.popleft(), alpha=0.5, label=roc["aux"] +
                 " EER: " + str(roc["eer"]))
    plt.xlabel("False Negative Rate")
    plt.ylabel("False Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
    plt.close()


def create_colours():
    colours = deque()
    colours.append("blue")
    colours.append("purple")
    colours.append("green")
    colours.append("brown")
    colours.append("yellow")
    return colours
