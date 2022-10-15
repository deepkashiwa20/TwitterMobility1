import numpy as np
import pandas as pd
from copy import copy
import torch
from torch import nn, Tensor
import matplotlib
# matplotlib.use('Agg')
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
    ax1.set_xticks([x for x in range(0, 12*2*8, 12)])
    ax1.set_xticklabels([])
    ax1.set_yticks([])
    ax2.imshow(att_without.transpose(), cmap=cmap, aspect='auto')
    ax2.set_ylabel('w/o', fontsize=15)
    ax2.set_xticks([x for x in range(0, 12*2*8, 12)])
    ax2.set_yticks([])

    dates = ['10-'+str(x) for x in range(8, 16)]
    labels = []
    for i in range(len(dates)):
        labels.append('')
        labels.append(dates[i])
    ax2.set_xticklabels(labels, fontsize=15)

    fig.set_size_inches(8, 1)
    plt.subplots_adjust(bottom=0.28, top=0.98, left=0.08, right=0.98)
    plt.show()
    # plt.savefig('./case_con_mem1.png')
    # then manually adjust the space using software
    
    return


if __name__ == '__main__':

    # w/ & w/o memory contrastive loss
    att_with = np.load(f'../model_ours/con_mem/typhoon-outflow_yes_202111092359/MemoryAGCRN_att_score.npy')
    att_without = np.load(f'../model_ours/con_mem/typhoon-outflow_no_202111161801/MemeSTN_att_score.npy')

    # test start: 10/06 10AM
    # +14h start: 10/07 0AM
    # 10/8~15: 8 days
    i, j = 9, 9
    att_with_i = att_with[(14+24):(14+24*9), i:(i+1)]     # 7, 8
    att_without_j = att_without[(14+24):(14+24*9), j:(j+1)]   # 3, 7, 9
    plot_con_mem(att_with_i, att_without_j)
