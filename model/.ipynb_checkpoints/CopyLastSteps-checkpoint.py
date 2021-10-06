import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import time
import sys
import shutil
import pandas as pd
import numpy as np
from Utils import get_pref_id, get_flow, get_adj, get_twitter, get_onehottime, get_data, get_seq_data
import Metrics

def CopyLastSteps(XS, YS):
    return XS

def testModel(name, mode, XS, YS):
    print('TIMESTEP_IN, TIMESTEP_OUT', opt.his_len, opt.seq_len)
    YS_pred = CopyLastSteps(XS, YS)
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    np.save(path + f'/{name}_prediction.npy', YS_pred)
    np.save(path + f'/{name}_groundtruth.npy', YS)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    print('*' * 40)
    f = open(path + f'/{name}_prediction_scores.txt', 'a')
    print("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (name, mode, MSE, RMSE, MAE, MAPE))
    f.write("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    for i in range(opt.seq_len):
        MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS[:, i, ...], YS_pred[:, i, ...])
        print("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
        f.write("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
    f.close()

##############  Typhoon-Inflow ###############
event = 'Typhoon'
flow_type = 'inflow'
flow_path = f'../data/{flow_type}_hour20180101_20210228.npy'
adj_path = '../data/adjacency_matrix.npy'
twitter_path = '../data/Japan_2019Hurricane_Total_tweet_count.csv'
pref_path = '../data/Japan_prefectures.csv'
freq = '1H'
flow_start_date, flow_end_date = '2018-01-01 00:00:00', '2021-02-28 23:59:59'
twitter_start_date, twitter_end_date = '2019-06-30 09:00:00', '2019-10-31 08:00:00'
target_start_date, target_end_date = '2019-07-01 00:00:00', '2019-10-30 23:00:00' # 2019-10-31 data is missing.
target_area = ['Fukushima', 'Ibaraki', 'Tochigi', 'Gunma', 'Saitama', 'Chiba', 'Tokyo', 'Kanagawa']
target_area_jp = ['福島県', '茨城県', '栃木県', '群馬県', '埼玉県', '千葉県', '東京都', '神奈川県']
tw_condition, his_condition = False, False
##############  Typhoon-Inflow ###############

##############  COVID-Inflow ###############
# event = 'COVID'
# flow_type = 'inflow'
# flow_path = f'../data/{flow_type}_hour20180101_20210228.npy'
# adj_path = '../data/adjacency_matrix.npy'
# twitter_path = '../data/Japan_COVID-19_Total_tweet_count.csv'
# pref_path = '../data/Japan_prefectures.csv'
# freq = '1H'
# flow_start_date, flow_end_date = '2018-01-01 00:00:00', '2021-02-28 23:59:59'
# twitter_start_date, twitter_end_date = '2019-12-31 09:00:00', '2021-02-28 08:00:00'
# target_start_date, target_end_date = '2020-01-01 00:00:00', '2020-12-31 23:00:00' # 2019-10-31 data is missing.
# target_area = ['Fukushima', 'Ibaraki', 'Tochigi', 'Gunma', 'Saitama', 'Chiba', 'Tokyo', 'Kanagawa']
# # target_area_jp = ['福島県', '茨城県', '栃木県', '群馬県', '埼玉県', '千葉県', '東京都', '神奈川県']
# tw_condition, his_condition = False, False
##############  COVID-Inflow ###############

model_name = 'CopyLastFrames'
path = f'./save/{event}_{flow_type}_{model_name}_' + time.strftime('%Y%m%d%H%M', time.localtime())

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=1000, help="number of epochs of training") # original 1500
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
parser.add_argument('--seq_len', type=int, default=12, help='sequence length of values, which should be even nums (2,4,6,12)')
parser.add_argument('--his_len', type=int, default=12, help='sequence length of observed historical values')
parser.add_argument('--num_variable', type=int, default=len(target_area), help='total number of the target variables') # current 7
parser.add_argument('--channel', type=int, default=1, help='channel')
parser.add_argument('--cond_feat', type=int, default=32 + sum([tw_condition, his_condition]), help='condition features of D and G')
parser.add_argument('--cond_source', type=int, default=sum([1, tw_condition, his_condition]), help='1 is only time label, 2 is his_x or twitter label, 3 is time, twitter, his')
parser.add_argument('--trainval_ratio', type=float, default=0.8, help='the total ratio of training data and validation data')
parser.add_argument('--val_ratio', type=float, default=0.2, help='the ratio of validation data among the trainval ratio')
parser.add_argument('--gpu', type=int, default=3, help='which gpu to use')
opt = parser.parse_args()

def main():
    if not os.path.exists(path):
        os.makedirs(path)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, path)
    
    # device = torch.device("cuda:{}".format(opt.gpu)) if torch.cuda.is_available() else torch.device("cpu")
    # np.random.seed(opt.seed)
    # torch.manual_seed(opt.seed)
    # if torch.cuda.is_available():
    #    torch.cuda.manual_seed(opt.seed)
    
    flow_all_times = [date.strftime('%Y-%m-%d %H:%M:%S') for date in pd.date_range(start=flow_start_date, end=flow_end_date, freq=freq)]
    start_index, end_index = flow_all_times.index(target_start_date), flow_all_times.index(target_end_date)
    area_index = get_pref_id(pref_path, target_area)
    flow = get_flow(flow_type, flow_path, start_index, end_index, area_index)
    print('original flow data ...', flow.shape, flow.min(), flow.max())
    onehottime = get_onehottime(target_start_date, target_end_date, freq)
    twitter = get_twitter(twitter_path, pref_path, target_start_date, target_end_date, target_area)
    adj = get_adj(adj_path, area_index) # it's already been normalized..
    x, c, tw, adj = get_data(flow, onehottime, twitter, adj, opt.num_variable, opt.channel)
    seq_x, seq_c, seq_tw, seq_adj = get_seq_data(x, opt.seq_len+opt.his_len), get_seq_data(c, opt.seq_len+opt.his_len), \
                                    get_seq_data(tw, opt.seq_len+opt.his_len), get_seq_data(adj, opt.seq_len+opt.his_len)
    print(flow.shape, twitter.shape, onehottime.shape, adj.shape)
    print(x.shape, c.shape, tw.shape, adj.shape)
    print(seq_x.shape, seq_c.shape, seq_tw.shape, seq_adj.shape)
    
    his_x, seq_x = seq_x[:, :opt.his_len, ...], seq_x[:, -opt.seq_len:, ...]
    his_c, seq_c = seq_c[:, :opt.his_len, ...], seq_c[:, -opt.seq_len:, ...] 
    his_tw, seq_tw = seq_tw[:, :opt.his_len, ...], seq_tw[:, -opt.seq_len:, ...]
    his_adj, seq_adj = seq_adj[:, :opt.his_len, ...], seq_adj[:, -opt.seq_len:, ...]
    print(his_x.shape, seq_x.shape, his_x.min(), his_x.max(), seq_x.min(), seq_x.max())
    print(his_c.shape, seq_c.shape, his_c.min(), his_c.max(), seq_c.min(), seq_c.max()) 
    print(his_tw.shape, seq_tw.shape, his_tw.min(), his_tw.max(), seq_tw.min(), seq_tw.max()) 
    print(his_adj.shape, seq_adj.shape, his_adj.min(), his_adj.max(), seq_adj.min(), seq_adj.max())
    
    num_train_sample = int(seq_x.shape[0] * opt.trainval_ratio)
    train_his_x, train_seq_x = his_x[:num_train_sample, ...], seq_x[:num_train_sample, ...]
    test_his_x, test_seq_x = his_x[num_train_sample:, ...], seq_x[num_train_sample:, ...]
    print(train_his_x.shape, train_seq_x.shape, test_his_x.shape, test_seq_x.shape)
    
    start = time.ctime()
    testModel(model_name, 'test', test_his_x, test_seq_x)
    end = time.ctime()
    print(event, flow_type, model_name, 'start and end time ...', start, end)
    
if __name__ == '__main__':
    main()
