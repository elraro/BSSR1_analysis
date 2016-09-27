import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import sys


def calculate_roc(matrix_dict):
    matrix = matrix_dict["matrix"]
    aux = matrix_dict["aux"]
    identity = np.identity(len(matrix))
    tp_rate = np.empty(shape=0)
    tn_rate = np.empty(shape=0)
    fn_rate = np.empty(shape=0)
    fp_rate = np.empty(shape=0)
    err_found = False
    err = 0
    for umbral in np.arange(0, 1.01, 0.01):
        # http://notmatthancock.github.io/2015/08/19/roc-curve-part-2-numerical-example.html
        tp = np.logical_and(matrix >= umbral, identity == 1).sum()
        tn = np.logical_and(matrix < umbral, identity == 0).sum()
        fp = np.logical_and(matrix >= umbral, identity == 0).sum()
        fn = np.logical_and(matrix < umbral, identity == 1).sum()
        tp_rate = np.append(tp_rate, tp / (tp + fn))
        tn_rate = np.append(tn_rate, tn / (tn + fp))
        fn_rate = np.append(fn_rate, fn / (fn + + tp))
        fp_rate = np.append(fp_rate, fp / (fp + tn))
        fn_rate_err = fn / (fn + + tp)
        fp_rate_err = fp / (fp + tn)
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
        plt.plot(roc["fn_rate"], roc["fp_rate"], linewidth=1, color=colours.popleft(), alpha=0.5,
                 label=roc["aux"] + " EER: " + str(roc["eer"]))
    plt.xlabel("False Negative Rate")
    plt.ylabel("False Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(title + ".png")
    plt.close()


def create_colours():
    colours = deque()
    colours.append("blue")
    colours.append("purple")
    colours.append("green")
    colours.append("brown")
    colours.append("yellow")
    return colours


def draw_roc_eer(roc_values, title, eer_1, eer_2):
    alpha = sys.maxsize
    eer = sys.maxsize
    plt.figure(figsize=(20, 10))  # figsize=(20,10) para aumentar el tamaño de la figura
    plt.ylim([0, 0.05])
    plt.hlines(eer_1, 0, 1, linestyle="dashed", color="red", linewidth=1, label="EER=" + str(eer_1))
    plt.hlines(eer_2, 0, 1, linestyle="dashed", color="blue", linewidth=1, label="EER=" + str(eer_2))
    eer_x = list()
    eer_y = list()
    for roc in roc_values:
        eer_x.append(roc["aux"])
        eer_y.append(roc["eer"])
        if roc["eer"] < eer:
            eer = roc["eer"]
            alpha = roc["aux"]
    plt.plot(eer_x, eer_y, linewidth=1, color="green", alpha=0.5, label="EER=" + str(eer) + " alpha=" + str(alpha))
    plt.xlabel("alpha")
    plt.ylabel("EER")
    plt.title(title)
    plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
    # plt.show()
    plt.savefig(title + ".png")
    plt.close()
    return alpha


def draw_roc_alphas(alpha_1, alpha_2, alpha_3, eer_1_list, eer_2_list, eer_3_list, text_1, text_2, text_3, title):
    for eer in eer_1_list:
        if eer["aux"] == alpha_1:
            eer_1 = eer
            break
    for eer in eer_2_list:
        if eer["aux"] == alpha_2:
            eer_2 = eer
            break
    for eer in eer_3_list:
        if eer["aux"] == alpha_3:
            eer_3 = eer
            break
    colours = create_colours()
    plt.figure()
    x = [0, 1]
    plt.plot(x, x, linestyle="dashed", color="red", linewidth=1)
    #for roc in roc_values:
    plt.plot(eer_1["fn_rate"], eer_1["fp_rate"], linewidth=1, color=colours.popleft(), alpha=0.5,
             label=text_1 + "alpha=" + str(eer_1["aux"]) + " eer=" + str(eer_1["eer"]))
    plt.plot(eer_2["fn_rate"], eer_2["fp_rate"], linewidth=1, color=colours.popleft(), alpha=0.5,
             label=text_2 + "alpha=" + str(eer_2["aux"]) + " eer=" + str(eer_2["eer"]))
    plt.plot(eer_3["fn_rate"], eer_3["fp_rate"], linewidth=1, color=colours.popleft(), alpha=0.5,
             label=text_3 + "alpha=" + str(eer_3["aux"]) + " eer=" + str(eer_3["eer"]))
    plt.xlabel("False Negative Rate")
    plt.ylabel("False Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(title + ".png")
    plt.close()
    return None