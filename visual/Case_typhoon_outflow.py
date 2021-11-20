import numpy as np
import pandas as pd
from copy import copy
import torch
from torch import nn, Tensor
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
#sns.axes_style("dark")



def plot_con_mem(att_with:np.array, att_without:np.array):
    print(att_with.shape, att_without.shape)

    cmap = copy(plt.cm.plasma)
    cmap.set_bad(cmap(0))

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.imshow(att_with.transpose(), cmap=cmap, aspect='auto')
    ax1.set_ylabel('w/', fontsize=15)
    ax1.set_xticks([x for x in range(0, 12*2*14, 12)])
    ax1.set_yticks([])
    ax2.imshow(att_without.transpose(), cmap=cmap, aspect='auto')
    ax2.set_ylabel('w/o', fontsize=15)
    ax2.set_xticks([x for x in range(0, 12*2*14, 12)])
    ax2.set_yticks([])

    dates = ['10-'+str(x).zfill(2) for x in range(7, 21)]
    labels = []
    for i in range(len(dates)):
        labels.append('')
        labels.append(dates[i])
    ax2.set_xticklabels(labels, fontsize=13)

    fig.set_size_inches(20, 3)
    plt.subplots_adjust(bottom=0.3, top=0.7)
    plt.show()

    return

if __name__ == '__main__':

    # w/ & w/o memory contrastive loss
    att_with = np.load(f'../model_ours/con_mem/covid-outflow_yes_202111101215/MemoryAGCRN_att_score.npy')
    att_without = np.load(f'../model_ours/con_mem/covid-outflow_no_202111161844/MemeSTN_att_score.npy')

    # test start: 10/06 10AM
    # +14h start: 10/07 0AM
    att_with = att_with[14:14+24*14, 4:5]
    att_without = att_without[14:14+24*14, 3:4]
    plot_con_mem(att_with, att_without)
