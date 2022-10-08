import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


def csv2mtx(distance_df:pd.DataFrame):
    num_county = len(fl_counties)
    adj = np.zeros((num_county, num_county))

    for i in range(distance_df.shape[0]):
        o = distance_df.iloc[i, 0]
        d = distance_df.iloc[i, 1]
        adj[fl_counties.index(o), fl_counties.index(d)] = distance_df.iloc[i, 2]

    return adj

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    print(np.allclose(a, a.T, rtol=rtol, atol=atol))
    return


fl_counties = ['Alachua', 'Baker', 'Bay', 'Bradford', 'Brevard', 'Broward', 'Calhoun', 'Charlotte', 'Citrus', 'Clay',
              'Collier', 'Columbia', 'DeSoto', 'Dixie', 'Duval', 'Escambia', 'Flagler', 'Franklin', 'Gadsden', 'Gilchrist',
              'Glades', 'Gulf', 'Hamilton', 'Hardee', 'Hendry', 'Hernando', 'Highlands', 'Hillsborough', 'Holmes',
              'Indian River', 'Jackson', 'Jefferson', 'Lafayette', 'Lake', 'Lee', 'Leon', 'Levy', 'Liberty', 'Madison',
              'Manatee', 'Marion', 'Martin', 'Miami-Dade', 'Monroe', 'Nassau', 'Okaloosa', 'Okeechobee', 'Orange', 'Osceola',
              'Palm Beach', 'Pasco', 'Pinellas', 'Polk', 'Putnam', 'Santa Rosa', 'Sarasota', 'Seminole', 'St. Johns', 'St. Lucie',
              'Sumter', 'Suwannee', 'Taylor', 'Union', 'Volusia', 'Wakulla', 'Walton', 'Washington']

if __name__ == '__main__':
    npz_data = np.load('./Hurricane-FL-20190601-20191030.npz')
    poi, tcov = npz_data['poi'], npz_data['tcov']
    print('POI visit:', poi.shape, 'Tcov:', tcov.shape)

    twitter = pd.read_csv('./US_2019Hurricane_FL_all_tweet_count_by_county.csv')
    print(twitter.head())
    print(twitter.tail())

    county_list = twitter.columns.to_list()[1:]
    fl_county = pd.DataFrame(columns=['prefecture_code', 'prefecture_en'])
    for i in range(len(county_list)):
        fl_county = fl_county.append({'prefecture_code':i+1, 'prefecture_en':county_list[i]}, ignore_index=True)
    #fl_county.to_csv('./Florida_counties.csv')

    twitter = twitter.iloc[5+24*3:-24-19, 1:].to_numpy()    # 2019/6/1~10/30
    print('Twitter:', twitter.shape)

    # save poi visit
    # np.savez_compressed('./Hurricane-FL.npz', poi=poi, tcov=tcov, twit=twitter)
    poi_1 = poi.sum(axis=-1)
    print('POI visit sum:', poi_1.shape)
    #np.save('./poi_hour20190601_20191030.npy', poi_1)


    # inv_dis mtx
    distance_df = pd.read_csv('./FL_county_distance_matrix.csv')
    print(distance_df.shape)
    dis = csv2mtx(distance_df)
    print(dis.shape)

    # check if inv_dis mtx is symmetric
    check_symmetric(dis)

    epsilon = 2e5
    inv_dis = np.exp(-(dis / epsilon) ** 2).astype('float32')   # Gaussian kernel
    inv_dis -= np.eye(inv_dis.shape[0])
    # plt.hist(adj.flatten())
    # plt.title('Adj')
    # plt.show()
    print(inv_dis)
    # np.save('./FL_county_inv_dis.npy', adj)


    # adj mtx
    fl_county_df = gpd.read_file('./geojson-fl-counties-fips.json')     # read county geojson
    fl_county_df = fl_county_df.sort_values(by=['id'], ignore_index=True)

    fl_county_df['adj_county'] = fl_county_df.apply(lambda x:np.array(fl_counties)[fl_county_df.geometry.touches(x['geometry'])], axis=1)

    print(fl_county_df.shape)
    print(fl_county_df)

    adj_mtx = np.zeros((len(fl_counties), len(fl_counties)), dtype='int32')
    for i in range(len(fl_counties)):
        adj_county_idx = fl_county_df.geometry.touches(fl_county_df.geometry.iloc[i])
        adj_mtx[i] = adj_county_idx

    print(adj_mtx)
    print(adj_mtx.shape)
    check_symmetric(adj_mtx)

    np.save('./FL_county_adj_mtx.npy', adj_mtx)
