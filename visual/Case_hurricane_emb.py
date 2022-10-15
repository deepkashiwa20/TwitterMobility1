import numpy as np
import pandas as pd
from sklearn import manifold
from copy import copy
import torch
from torch import nn, Tensor
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

    # Hurricane POI
    # w/ & w/o & noise memory contrastive loss
    with_train = np.load(f'../save_hurricane-poi/hurricane-poi_MemoryAGCRN_20221011013850_paper_lamb0.1/MemoryAGCRN_train_memories.npz')
    with_test = np.load(f'../save_hurricane-poi/hurricane-poi_MemoryAGCRN_20221011013850_paper_lamb0.1/MemoryAGCRN_test_memories.npz')

    without_train = np.load(f'../save_hurricane-poi/hurricane-poi_MemoryAGCRN_20221011013927_good_lamb0.0/MemoryAGCRN_train_memories.npz')
    without_test = np.load(f'../save_hurricane-poi/hurricane-poi_MemoryAGCRN_20221011013927_good_lamb0.0/MemoryAGCRN_test_memories.npz')

    noise_train = np.load(f'../save_hurricane-poi/hurricane-poi_MemeSTNnoise_20221011152205_paper/MemeSTNnoise_train_memories.npz')
    noise_test = np.load(f'../save_hurricane-poi/hurricane-poi_MemeSTNnoise_20221011152205_paper/MemeSTNnoise_test_memories.npz')

    # for key in with_train.keys():
    #     print(key, with_train[key].shape)   # (1373,
    # for key in with_test.keys():
    #     print(key, with_test[key].shape)    # (344,
    # for key in without_train.keys():
    #     print(key, without_train[key].shape)
    # for key in without_test.keys():
    #     print(key, without_test[key].shape)
    # for key in noise_train.keys():
    #     print(key, noise_train[key].shape)
    # for key in noise_test.keys():
    #     print(key, noise_test[key].shape)

    with_query = np.concatenate([with_train['query'], with_test['query']], axis=0)
    with_att = np.concatenate([with_train['att'], with_test['att']], axis=0)
    with_proto = np.concatenate([with_train['proto'], with_test['proto']], axis=0)
    with_temb = np.concatenate([with_train['emb'], with_test['emb']], axis=0)
    with_mem = with_train['memo']

    without_query = np.concatenate([without_train['query'], without_test['query']], axis=0)
    without_att = np.concatenate([without_train['att'], without_test['att']], axis=0)
    without_proto = np.concatenate([without_train['proto'], without_test['proto']], axis=0)
    without_temb = np.concatenate([without_train['emb'], without_test['emb']], axis=0)
    without_mem = without_train['memo']

    noise_query = np.concatenate([noise_train['query'], noise_test['query']], axis=0)
    noise_att = np.concatenate([noise_train['att'], noise_test['att']], axis=0)
    noise_proto = np.concatenate([noise_train['proto'], noise_test['proto']], axis=0)
    noise_temb = np.concatenate([noise_train['emb'], noise_test['emb']], axis=0)
    noise_mem = noise_train['memo']

    # hurricane index
    hurricane_index = [range(1512, 1560)]       # Dorian 3-day 9/1~3

    # instantiate tSNE
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)

    Q_P = np.concatenate([with_proto, noise_proto], axis=0)
    emb = tsne.fit_transform(Q_P)

    emb_min, emb_max = emb.min(0), emb.max(0)
    emb_norm = (emb - emb_min) / (emb_max - emb_min)

    emb_list = np.split(emb_norm, 2, axis=0)
    Qemb, Oemb = emb_list[0], emb_list[1]

    # plot
    fig, ax = plt.subplots()

    ax.scatter(np.delete(Qemb[:, 0], hurricane_index), np.delete(Qemb[:, 1], hurricane_index), marker='.', color='royalblue')
    ax.scatter(Qemb[hurricane_index, 0], Qemb[hurricane_index, 1], marker='x', color='navy')
    #ax.scatter(np.delete(Pemb[:, 0], hurricane_index), np.delete(Pemb[:, 1], hurricane_index), marker='.', color='tab:orange')
    #ax.scatter(Pemb[hurricane_index, 0], Pemb[hurricane_index, 1], marker='x', color='tab:red')
    ax.scatter(np.delete(Oemb[:, 0], hurricane_index), np.delete(Oemb[:, 1], hurricane_index), marker='.', color='limegreen')
    ax.scatter(Oemb[hurricane_index, 0], Oemb[hurricane_index, 1], marker='x', color='darkgreen')

    legend_elements = [
        Line2D([0], [0], marker='o', color='white', label='Socio: normal', markerfacecolor='royalblue', markersize=8, alpha=0.8),
        Line2D([0], [0], marker='X', color='white', label='Socio: hurricane', markerfacecolor='navy', markersize=8, alpha=0.8),
        Line2D([0], [0], marker='o', color='white', label='', markerfacecolor='white'),
        Line2D([0], [0], marker='o', color='white', label='Noise: normal', markerfacecolor='limegreen', markersize=8, alpha=0.8),
        Line2D([0], [0], marker='X', color='white', label='Noise: hurricane', markerfacecolor='darkgreen', markersize=8, alpha=0.8)]
    ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.9, 0.35, 0.5, 0.5), fontsize=13, frameon=False)

    fig.set_size_inches(8, 4)
    plt.subplots_adjust(bottom=0, top=1, left=0, right=0.7)
    plt.axis('off')
    plt.show()
