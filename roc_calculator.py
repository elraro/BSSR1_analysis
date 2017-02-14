import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import sys


def calculate_eer_train(matrix_dict):
    matrix = matrix_dict["matrix"]
    aux = matrix_dict["aux"]
    identity = np.identity(len(matrix))
    tp_rate = np.empty(shape=0)
    tn_rate = np.empty(shape=0)
    fn_rate = np.empty(shape=0)
    fp_rate = np.empty(shape=0)
    eer_dif = sys.maxsize
    eer = 0
    for umbral in np.arange(0, 1.01, 0.01):
        tp = np.logical_and(matrix >= umbral, identity == 1).sum()
        tn = np.logical_and(matrix < umbral, identity == 0).sum()
        fp = np.logical_and(matrix >= umbral, identity == 0).sum()
        fn = np.logical_and(matrix < umbral, identity == 1).sum()
        tp_rate = np.append(tp_rate, tp / (tp + fn))
        tn_rate = np.append(tn_rate, tn / (tn + fp))
        fn_rate = np.append(fn_rate, fn / (fn + + tp))
        fp_rate = np.append(fp_rate, fp / (fp + tn))
        fn_rate_eer = fn / (fn + tp)
        fp_rate_eer = fp / (fp + tn)
        if abs(fp_rate_eer - fn_rate_eer) < eer_dif:
            eer_dif = abs(fp_rate_eer - fn_rate_eer)
            eer = (fp_rate_eer + fn_rate_eer) / 2
    roc = dict()
    roc["tp_rate"] = tp_rate
    roc["tn_rate"] = tn_rate
    roc["fn_rate"] = fn_rate
    roc["fp_rate"] = fp_rate
    roc["eer"] = eer
    roc["aux"] = aux
    return roc


def calculate_alpha(eer_values):
    alpha = sys.maxsize
    eer = sys.maxsize
    for eer_aux in eer_values:
        if eer_aux["eer"] < eer:
            eer = eer_aux["eer"]
            alpha = eer_aux["aux"]
    return alpha


def draw_roc(roc_values, title, title_fusion=""):
    colours = create_colours()
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'cm'
    plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0
    plt.figure()
    x = [0, 1]
    plt.plot(x, x, linestyle="dashed", color="red", linewidth=1)
    for roc in roc_values:
        if type(roc["aux"]) != str:
            plt.plot(roc["fn_rate"], roc["fp_rate"], linewidth=1, color=colours.popleft(), alpha=0.5,
                     label=title_fusion + " a=" + str(roc["aux"]) + " EER=" + str("{0:.4f}".format(roc["eer"])))
        else:
            plt.plot(roc["fn_rate"], roc["fp_rate"], linewidth=1, color=colours.popleft(), alpha=0.5,
                     label=roc["aux"] + " EER=" + str("{0:.4f}".format(roc["eer"])))
    plt.xlabel("False Negative Rate")
    plt.ylabel("False Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(title + ".png")
    plt.close()


def draw_compare_eer(roc_values, title, eer_1, eer_2, label_eer_1, label_eer_2):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'cm'
    plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0
    plt.figure(figsize=(20, 10))  # figsize=(20,10) para aumentar el tamaño de la figura
    plt.ylim([0, 0.05])
    plt.hlines(eer_1, 0, 1, linestyle="dashed", color="red", linewidth=1, label="EER=" + label_eer_1)
    plt.hlines(eer_2, 0, 1, linestyle="dashed", color="blue", linewidth=1, label="EER=" + label_eer_2)
    eer_x = list()
    eer_y = list()
    for roc in roc_values:
        eer_x.append(roc["aux"])
        eer_y.append(roc["eer"])
    plt.plot(eer_x, eer_y, linewidth=1, color="green", alpha=0.5, label="EER fusion")
    plt.xlabel("alpha")
    plt.ylabel("EER")
    plt.title(title)
    plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
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
