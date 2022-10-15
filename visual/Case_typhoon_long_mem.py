import numpy as np
import pandas as pd
from copy import copy
import torch
from torch import nn, Tensor
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
#sns.axes_style("dark")



def plot_con_mem(att_with:np.array, att_without:np.array, att_noise:np.array, time_slice:int):
    t_interval = int(24//time_slice)

    cmap = copy(plt.cm.jet)
    cmap.set_bad(cmap(0))
    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.imshow(att_with.transpose(), cmap=cmap, aspect='auto')
    ax1.set_ylabel('social', fontsize=16)
    ax1.set_xticks([0, t_interval*15, t_interval*31, t_interval*(31+15), t_interval*62, t_interval*(62+15), t_interval*92, t_interval*(92+15)])
    ax1.set_xticklabels([])
    ax1.set_yticks([0,3,6])

    ax2.imshow(att_noise.transpose(), cmap=cmap, aspect='auto')
    ax2.set_ylabel('noise', fontsize=16)
    ax2.set_xticks([0, t_interval*15, t_interval*31, t_interval*(31+15), t_interval*62, t_interval*(62+15), t_interval*92, t_interval*(92+15)])
    ax2.set_xticklabels([])
    ax2.set_yticks([0,3,6])
    ax2.set_xticklabels(['Jul-1', 'Jul-15', 'Aug-1', 'Aug-15', 'Sep-1', 'Sep-15', 'Oct-1', 'Oct-15'], fontsize=15)

    fig.set_size_inches(8, 2)
    plt.subplots_adjust(bottom=0.16, top=0.98, left=0.08, right=0.98)
    plt.show()

    return


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

    # test start: 10/06 10AM?
    # +14h start: 10/07 0AM?
    # att_with = with_test['att'][14:14 + 24 * 14, :]
    # att_without = without_test['att'][14:14 + 24 * 14, :]
    # att_noise = noise_test['att'][14:14 + 24 * 14, :]

    # todo: train:test = 2333:584 -> test start: 10/06 5AM!
    time_slice = 6

    with_att = temporal_consecutive_aggregate(with_att[:-(24-11)], time_slice)
    without_att = temporal_consecutive_aggregate(without_att[:-(24 - 11)], time_slice)
    noise_att = temporal_consecutive_aggregate(noise_att[:-(24 - 11)], time_slice)
    print(with_att.shape, without_att.shape, noise_att.shape)  # 121 days

    with_att = np.concatenate([with_att[:,:4], with_att[:,6:]], axis=-1)
    without_att = np.concatenate([with_att[:,:5], with_att[:,6:]], axis=-1)
    noise_att = np.concatenate([noise_att[:, :2], noise_att[:, 4:]], axis=-1)
    plot_con_mem(with_att, without_att, noise_att, time_slice)
