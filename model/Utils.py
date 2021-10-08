import pandas as pd
import numpy as np
import scipy.sparse as ss
import jpholiday

def get_flow(flow_type, flow_path, start_index, end_index, area_index):
    assert flow_type in flow_path, "Please check if the flow data is compatible with flow type."
    if flow_type == 'odflow':
        odflow = ss.load_npz(flow_path)
        flow = np.array(flow.todense()).reshape((-1, 47, 47))
        flow_pad = np.zeros((flow.shape[0], flow.shape[1] + 1, flow.shape[2] + 1))
        flow_pad[:, :flow.shape[1], :flow.shape[2]] = flow
        flow_pad = flow_pad[start_index:end_index+1, :, :]
        flow_pad = flow_pad[:, area_index, :][:, :, area_index]
    else:
        flow = np.load(flow_path)
        flow_pad = np.zeros((flow.shape[0], flow.shape[1] + 1))
        flow_pad[:, :flow.shape[1]] = flow
        flow_pad = flow_pad[start_index:end_index + 1, :]
        flow_pad = flow_pad[:, area_index]
    return flow_pad

def get_twitter(twitter_path, pref_path, start_date, end_date, area_list):
    twitter = pd.read_csv(twitter_path, index_col=0)
    twitter = twitter[area_list]
    twitter = twitter.loc[start_date:end_date]
    return twitter.values

def get_onehottime(start_date, end_date, freq):
    df = pd.DataFrame({'time': pd.date_range(start_date, end_date, freq=freq)})
    df['dayofweek'] = df.time.dt.weekday
    df['hourofday'] = df.time.dt.hour
    df['isholiday'] = df.apply(lambda x: int(jpholiday.is_holiday(x.time) | (x.dayofweek==5) | (x.dayofweek==6)), axis=1)
    tmp1 = pd.get_dummies(df.dayofweek)
    tmp2 = pd.get_dummies(df.hourofday)
    tmp3 = df[['isholiday']]
    df_dummy = pd.concat([tmp1, tmp2, tmp3], axis=1)
    return df_dummy.values

def get_adj(adj_path, area_index):
    def sym_adj(adj):
        """Symmetrically normalize adjacency matrix."""
        adj = ss.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = ss.diags(d_inv_sqrt)
        return np.array(adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense())

    adj = np.load(adj_path)
    adj_pad = np.zeros((adj.shape[0] + 1, adj.shape[1] + 1))
    adj_pad[:adj.shape[0], :adj.shape[1]] = adj
    np.fill_diagonal(adj_pad, 1)
    adj_pad = adj_pad[area_index, :][:, area_index]
    sym_adj = sym_adj(adj_pad)
    return sym_adj

def get_pref_id(pref_path, target_pref):
    jp_pref = pd.read_csv(pref_path, index_col=2)
    if len(target_pref) == 0 or target_pref == None:
        return (jp_pref['prefecture_code'] - 1).values.tolist()
    else:
        return (jp_pref.loc[target_pref]['prefecture_code'] - 1).values.tolist()
    
# min-max normalization: x -> [-1,1]
def min_max_normal(x):
    max_x = x.max()
    min_x = x.min()
    x = (x - min_x) / (max_x - min_x)
    x = 2.0*x - 1.0
    return x # return x, min_x, max_x if we want to revert transform

def get_data(x, cond, tw, adj, num_variable, channel):  
    x = x[:, :, np.newaxis].repeat(channel, axis=2) # final_feat=channel=1
    tw = tw[:, :, np.newaxis].repeat(channel, axis=2) # final_feat=channel=1
    cond = cond[:, np.newaxis, ].repeat(num_variable, axis=1) # this is condition/label
    adj = adj[np.newaxis, :, :].repeat(x.shape[0], axis=0)
    return x, cond, tw, adj

def get_seq_data(data, seq_len):
    seq_data = [data[i:i+seq_len, ...] for i in range(0, data.shape[0]-seq_len+1)]
    return np.array(seq_data)