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



def temporal_consecutive_aggregate(x:np.array, time_slice:int):
    assert x.shape[0] % time_slice == 0

    x_slice = []
    for i in range(time_slice):
        x_slice.append(x[i::time_slice,...])
    return sum(x_slice)


if __name__ == '__main__':

    # Typhoon outflow
    # w/ & w/o & noise memory contrastive loss
    with_train = np.load(f'../save_hurricane-poi/typhoon-outflow_yes_202111092359/MemoryAGCRN_train_memories.npz')
    with_test = np.load(f'../save_hurricane-poi/typhoon-outflow_yes_202111092359/MemoryAGCRN_test_memories.npz')

    without_train = np.load(f'../save_hurricane-poi/typhoon-outflow_no_202111161801/MemeSTN_train_memories.npz')
    without_test = np.load(f'../save_hurricane-poi/typhoon-outflow_no_202111161801/MemeSTN_test_memories.npz')

    noise_train = np.load(f'../save_hurricane-poi/typhoon-outflow_noise1_202204261154/MemeSTNnoise_train_memories.npz')
    noise_test = np.load(f'../save_hurricane-poi/typhoon-outflow_noise1_202204261154/MemeSTNnoise_test_memories.npz')

    # for key in with_train.keys():
    #     print(key, with_train[key].shape)   # (2333,
    # for key in with_test.keys():
    #     print(key, with_test[key].shape)    # (584,
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

    # typhoon index
    typhoon_index = [range(1092, 1116), range(2448, 2472), range(2472, 2496), range(2496, 2520)]      # Krosa 1-day, Hagibis 3-day

    # instantiate tSNE
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)

    Q_P = np.concatenate([with_proto, without_proto, noise_proto], axis=0)
    emb = tsne.fit_transform(Q_P)

    emb_min, emb_max = emb.min(0), emb.max(0)
    emb_norm = (emb - emb_min) / (emb_max - emb_min)

    emb_list = np.split(emb_norm, 3, axis=0)
    Qemb, Pemb, Oemb = emb_list[0], emb_list[1], emb_list[2]

    # plot
    fig, ax = plt.subplots()

    ax.scatter(np.delete(Qemb[:, 0], typhoon_index), np.delete(Qemb[:, 1], typhoon_index), marker='.', color='royalblue')
    ax.scatter(Qemb[typhoon_index, 0], Qemb[typhoon_index, 1], marker='x', color='navy')
    ax.scatter(np.delete(Pemb[:, 0], typhoon_index), np.delete(Pemb[:, 1], typhoon_index), marker='.', color='sandybrown')
    ax.scatter(Pemb[typhoon_index, 0], Pemb[typhoon_index, 1], marker='x', color='darkorange')
    ax.scatter(np.delete(Oemb[:, 0], typhoon_index), np.delete(Oemb[:, 1], typhoon_index), marker='.', color='limegreen')
    ax.scatter(Oemb[typhoon_index, 0], Oemb[typhoon_index, 1], marker='x', color='darkgreen')

    legend_elements = [
        Line2D([0], [0], marker='o', color='white', label='Socio w/ Lcon: normal', markerfacecolor='royalblue', markersize=8, alpha=0.8),
        Line2D([0], [0], marker='X', color='white', label='Socio w/ Lcon: typhoon', markerfacecolor='navy', markersize=8, alpha=0.8),
        Line2D([0], [0], marker='o', color='white', label='', markerfacecolor='white'),
        Line2D([0], [0], marker='o', color='white', label='Socio w/o Lcon: normal', markerfacecolor='sandybrown', markersize=8, alpha=0.8),
        Line2D([0], [0], marker='X', color='white', label='Socio w/o Lcon: typhoon', markerfacecolor='darkorange', markersize=8, alpha=0.8),
        Line2D([0], [0], marker='o', color='white', label='', markerfacecolor='white'),
        Line2D([0], [0], marker='o', color='white', label='Noise: normal', markerfacecolor='limegreen', markersize=8, alpha=0.8),
        Line2D([0], [0], marker='X', color='white', label='Noise: typhoon', markerfacecolor='darkgreen', markersize=8, alpha=0.8)]
    ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.95, 0.2, 0.5, 0.5), fontsize=12, frameon=False)

    fig.set_size_inches(8, 4)
    plt.subplots_adjust(bottom=0, top=1, left=0, right=0.7)
    plt.axis('off')
    plt.show()
