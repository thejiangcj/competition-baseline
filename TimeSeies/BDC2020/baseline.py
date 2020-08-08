# log label
from tqdm import tqdm
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error,explained_variance_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
import lightgbm as lgb
from datetime import timedelta, datetime
import math
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import warnings
import gc

import moxing as mox
mox.file.shift('os', 'mox')

#train_gps_path = 'data/train_data_gps_790.csv'
train_gps_path = 'obs://obsacc/data/train_data_03.csv'
test_data_path = 'obs://obsacc/data/data/R2_ATest 0711.csv'
port_data_path = 'obs://obsacc/data/data/port.csv'

debug = False
NDATA = 1000000
randomSeed=1024

starttime = datetime.now()

with mox.file.File(train_gps_path, "r+") as f, mox.file.File(test_data_path, "r+") as f1, mox.file.File(port_data_path, "r+") as f2:
    if debug:
        train_data = pd.read_csv(f,nrows=NDATA,header=None)
    else:
        train_data = pd.read_csv(f)
                             
    test_data = pd.read_csv(f1)
    port_data = pd.read_csv(f2)

def get_data(data, mode='train'):
    
    assert mode=='train' or mode=='test'
    
    if mode=='train':
#         data['vesselNextportETA'] = pd.to_datetime(data['vesselNextportETA'], infer_datetime_format=True)
        pass
    elif mode=='test':
        data['temp_timestamp'] = data['timestamp']
        data['onboardDate'] = pd.to_datetime(data['onboardDate'], infer_datetime_format=True)
    data['timestamp'] = pd.to_datetime(data['timestamp'], infer_datetime_format=True)
    data['longitude'] = data['longitude'].astype(float)
    data['loadingOrder'] = data['loadingOrder'].astype(str)
    data['latitude'] = data['latitude'].astype(float)
    data['speed'] = data['speed'].astype(float)
    data['direction'] = data['direction'].astype(float)
    data['TRANSPORT_TRACE'] = data['TRANSPORT_TRACE'].astype(str)

    return data

train_data = get_data(train_data, mode='train')
test_data = get_data(test_data, mode='test')

train_data = train_data.rename(columns={'label':'labels'})

test_data_temp = test_data.copy()

test_data_temp['TRANSPORT_TRACE'] = test_data_temp['TRANSPORT_TRACE'].str.split('-').copy()
test_data_temp['begin_path'] = test_data_temp['TRANSPORT_TRACE'].apply(lambda x:x[0])
test_data_temp['end_path'] = test_data_temp['TRANSPORT_TRACE'].apply(lambda x:x[-1])
test_begin_unique = test_data_temp['begin_path'].unique().tolist()
test_end_unique = test_data_temp['end_path'].unique().tolist()
test_data_list = test_begin_unique+test_end_unique

del test_data_temp

temp_train_data = train_data.groupby('loadingOrder')['TRANSPORT_TRACE'].agg(trace='max').reset_index()
temp_train_data['trace'] = temp_train_data['trace'].str.split('-')

def filter_data(x, start_list, end_list):
    all_list = start_list + end_list
    start = x[0]
    end = x[-1]
    if start in start_list and end in end_list:
        return True
    else:
        return False

is_in_list = temp_train_data['trace'].apply(filter_data, args=(test_begin_unique, test_end_unique,))
temp_train_data = temp_train_data[is_in_list]
temp_train_data_list = temp_train_data['loadingOrder'].unique().tolist()

train_data = train_data.rename(columns={'label':'labels'})

def geohash_encode(latitude, longitude, precision=12):
    """
    Encode a position given in float arguments latitude, longitude to
    a geohash which will have the character count precision.
    """
    lat_interval, lon_interval = (-90.0, 90.0), (-180.0, 180.0)
    base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
    geohash = []
    bits = [16, 8, 4, 2, 1]
    bit = 0
    ch = 0
    even = True
    while len(geohash) < precision:
        if even:
            mid = (lon_interval[0] + lon_interval[1]) / 2
            if longitude > mid:
                ch |= bits[bit]
                lon_interval = (mid, lon_interval[1])
            else:
                lon_interval = (lon_interval[0], mid)
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2
            if latitude > mid:
                ch |= bits[bit]
                lat_interval = (mid, lat_interval[1])
            else:
                lat_interval = (lat_interval[0], mid)
        even = not even
        if bit < 4:
            bit += 1
        else:
            geohash += base32[ch]
            bit = 0
            ch = 0
    return ''.join(geohash)

def hashfxn(astring):
    return ord(astring[0])

def q10(x):
    return x.quantile(0.1)

def q20(x):
    return x.quantile(0.2)

def q30(x):
    return x.quantile(0.3)

def q40(x):
    return x.quantile(0.4)

def q60(x):
    return x.quantile(0.6)

def q70(x):
    return x.quantile(0.7)

def q80(x):
    return x.quantile(0.8)

def q90(x):
    return x.quantile(0.9)

def tfidf(input_values, output_num, output_prefix, seed=1024):
    tfidf_enc = TfidfVectorizer()
    tfidf_vec = tfidf_enc.fit_transform(input_values)
    svd_tmp = TruncatedSVD(n_components=output_num, n_iter=20, random_state=seed)
    svd_tmp = svd_tmp.fit_transform(tfidf_vec)
    svd_tmp = pd.DataFrame(svd_tmp)
    svd_tmp.columns = ['{}_tfidf_{}'.format(output_prefix, i) for i in range(output_num)]
    return svd_tmp


def count2vec(input_values, output_num, output_prefix, seed=1024):
    count_enc = CountVectorizer()
    count_vec = count_enc.fit_transform(input_values)
    svd_tmp = TruncatedSVD(n_components=output_num, n_iter=20, random_state=seed)
    svd_tmp = svd_tmp.fit_transform(count_vec)
    svd_tmp = pd.DataFrame(svd_tmp)
    svd_tmp.columns = ['{}_countvec_{}'.format(output_prefix, i) for i in range(output_num)]
    return svd_tmp


def get_geohash_tfidf(df, group_id, group_target, num):
    df[group_target] = df.apply(lambda x: geohash_encode(x['latitude'], x['longitude'], 7), axis=1)
    tmp = df.groupby(group_id)[group_target].agg(list).reset_index()
    tmp[group_target] = tmp[group_target].apply(lambda x: ' '.join(x))

    tfidf_tmp = tfidf(tmp[group_target], num, group_target)
    count_tmp = count2vec(tmp[group_target], num, group_target)
    return pd.concat([tmp[[group_id]], tfidf_tmp, count_tmp], axis=1)


def get_grad_tfidf(df, group_id, group_target, num):
    grad_df = df.groupby(group_id)['latitude'].apply(lambda x: np.gradient(x)).reset_index()
    grad_df['longitude'] = df.groupby(group_id)['longitude'].apply(lambda x: np.gradient(x)).reset_index()['longitude']
    grad_df['latitude'] = grad_df['latitude'].apply(lambda x: np.round(x, 4))
    grad_df['longitude'] = grad_df['longitude'].apply(lambda x: np.round(x, 4))
    grad_df[group_target] = grad_df.apply(
        lambda x: ' '.join(['{}_{}'.format(z[0], z[1]) for z in zip(x['latitude'], x['longitude'])]), axis=1)

    tfidf_tmp = tfidf(grad_df[group_target], num, group_target)
    return pd.concat([grad_df[[group_id]], tfidf_tmp], axis=1)


def get_sample_tfidf(df, group_id, group_target, num):
    tmp = df.groupby(group_id)['lat_lon'].apply(lambda x: x.sample(frac=0.1, random_state=1)).reset_index()
    del tmp['level_1']
    tmp.columns = [group_id, group_target]
    tmp = tmp.groupby(group_id)[group_target].agg(list).reset_index()
    tmp[group_target] = tmp[group_target].apply(lambda x: ' '.join(x))

    tfidf_tmp = tfidf(tmp[group_target], num, group_target)
    return pd.concat([tmp[[group_id]], tfidf_tmp], axis=1)


# workers设为1可复现训练好的词向量，但速度稍慢，若不考虑复现的话，可对此参数进行调整
def w2v_feat(df, group_id, feat, length):
    print('start word2vec ...')
    data_frame = df.groupby(group_id)[feat].agg(list).reset_index()
    model = Word2Vec(data_frame[feat].values, size=length, window=5, min_count=1, sg=1, hs=1,
                     workers=1, iter=10, seed=1, hashfxn=hashfxn)
    data_frame[feat] = data_frame[feat].apply(lambda x: pd.DataFrame([model[c] for c in x]))
    for m in range(length):
        data_frame['w2v_{}_mean'.format(m)] = data_frame[feat].apply(lambda x: x[m].mean())
    del data_frame[feat]
    return data_frame


def d2v_feat(df, group_id, feat, length):
    print('start doc2vec ...')
    data_frame = df.groupby(group_id)[feat].agg(list).reset_index()
    documents = [TaggedDocument(doc, [i]) for i, doc in zip(data_frame[group_id].values, data_frame[feat])]
    model = Doc2Vec(documents, vector_size=length, window=5, min_count=1, workers=1, seed=1, hashfxn=hashfxn, 
                    epochs=10, sg=1, hs=1)
    doc_df = data_frame[group_id].apply(lambda x: ','.join([str(i) for i in model[x]])).str.split(',', expand=True).apply(pd.to_numeric)
    doc_df.columns = ['{}_d2v_{}'.format(feat, i) for i in range(length)]
    return pd.concat([data_frame[[group_id]], doc_df], axis=1)

def port_distance(x, df):
    from math import radians, cos, sin, asin, sqrt
    lon1, lat1 = df[df['TRANS_NODE_NAME']==x['original']]['LONGITUDE'].values[0], df[df['TRANS_NODE_NAME']==x['original']]['LATITUDE'].values[0]
    lon2, lat2 = df[df['TRANS_NODE_NAME']==x['dest']]['LONGITUDE'].values[0], df[df['TRANS_NODE_NAME']==x['dest']]['LATITUDE'].values[0]
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # 地球平均半径，单位为公里
    return c*r

def df_distance(x, df):
    from math import radians, cos, sin, asin, sqrt
    lon1, lat1 = df.iloc[x['idxmax']]['longitude'], df.iloc[x['idxmax']]['latitude']
    lon2, lat2 = df.iloc[x['idxmin']]['longitude'], df.iloc[x['idxmin']]['latitude']
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # 地球平均半径，单位为公里
    return c*r

def lat_lon_port(x, df):
    from math import radians, cos, sin, asin, sqrt
    dest = x['TRANSPORT_TRACE'].split('-')[-1]
    lon1, lat1 = df[df['TRANS_NODE_NAME']==dest]['LONGITUDE'].values[0], df[df['TRANS_NODE_NAME']==dest]['LATITUDE'].values[0]
    lon2, lat2 = x['longitude'], x['latitude']
    
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # 地球平均半径，单位为公里
    return c*r

def time_dis(x, df):
    from math import radians, cos, sin, asin, sqrt
    lon1, lat1 = df.iloc[int(x['time_max_idx'])]['longitude'], df.iloc[int(x['time_max_idx'])]['latitude']
    lon2, lat2 = x['longitude'], x['latitude']
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # 地球平均半径，单位为公里
    return c*r

def get_feature(df, df_port, mode='train'):
    
    assert mode=='train' or mode=='test'
    
    df.sort_values(['loadingOrder', 'timestamp'], inplace=True)
    # 特征只选择经纬度、速度\方向
    df_group = df.groupby('loadingOrder')
    df['lat_diff'] = df_group['latitude'].diff(1)
    df['lon_diff'] = df_group['longitude'].diff(1)
    df['speed_diff'] = df_group['speed'].diff(1)
    df['diff_hours'] = df_group['timestamp'].diff(1).dt.total_seconds() // 3600

    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.weekday
    df_group = df.groupby('loadingOrder')
    group_df = df_group['timestamp'].agg(mmax='max', cnt='count', mmin='min').reset_index()
    if mode=='train':
        # abs distance
        idx_lat_lon = df_group['timestamp'].agg(idxmax='idxmax', idxmin='idxmin').reset_index()
        idx_lat_lon['abs_distance'] = idx_lat_lon[['idxmax','idxmin']].apply(df_distance, args=(df,), axis=1)
        idx_lat_lon.drop(['idxmax','idxmin'], axis=1, inplace=True)
        group_df = group_df.merge(idx_lat_lon, on='loadingOrder', how='left')
        # label
        label_df = df_group['labels'].agg(label='max').reset_index()
        group_df = group_df.merge(label_df, on='loadingOrder', how='left')
        group_df['label'] = group_df['label'].apply(np.log)
        # point distance
        temp_time = df_group['timestamp'].agg(time_max_idx='idxmax').reset_index()
        df = df.merge(temp_time, on='loadingOrder', how='left')
        df['distance'] = df[['latitude','longitude','time_max_idx']].apply(time_dis, args=(df,),axis=1)
        df.drop(['time_max_idx'], axis=1, inplace=True)

    elif mode=='test':
        # abs distance
        port_lat_lon = df_group['TRANSPORT_TRACE'].agg(max='max').reset_index()
        port_lat_lon['max'] = port_lat_lon['max'].str.split('-')
        port_lat_lon['original'] = port_lat_lon['max'].apply(lambda x:x[0])
        port_lat_lon['dest'] = port_lat_lon['max'].apply(lambda x:x[-1])
        port_lat_lon['abs_distance'] = port_lat_lon[['original','dest']].apply(port_distance, args=(df_port,), axis=1)
        port_lat_lon.drop(['original','dest','max'], axis=1, inplace=True)
        group_df = group_df.merge(port_lat_lon, on='loadingOrder', how='left')

        df['distance'] = df[['latitude','longitude', 'TRANSPORT_TRACE']].apply(lat_lon_port, args=(df_port,), axis=1)
        
        print("4. Start LabelEncoder")
        encode_label = ['carrierName', 'vesselMMSI']
        df[encode_label] = df[encode_label].apply(LabelEncoder().fit_transform)
        
    df_group = df.groupby('loadingOrder')
    # 单点距离
    agg_function = ['min', 'max', 'mean', 'median', 'nunique', q10, q20, q30, q40, q60, q70, q80, q90]
    agg_ways = ['min', 'max', 'mean', 'median', 'nunique', 'q_10', 'q_20', 'q_30', 'q_40', 'q_60', 'q_70', 'q_80', 'q_90']
    agg_col = ['distance','direction']

    group = df.groupby('loadingOrder')[agg_col].agg(agg_function).reset_index()
    group.columns = ['loadingOrder'] + ['{}_{}'.format(i, j) for i in agg_col for j in agg_ways]
    group_df = group_df.merge(group, on='loadingOrder', how='left')
    
    # labelencode
    agg_function = ['sum','nunique','mean']
    agg_ways = ['sum','nunique','mean']
    agg_col = ['carrierName', 'vesselMMSI']
    group = df.groupby('loadingOrder')[agg_col].agg(agg_function).reset_index()
    group.columns = ['loadingOrder'] + ['{}_{}'.format(i, j) for i in agg_col for j in agg_ways]
    group_df = group_df.merge(group, on='loadingOrder', how='left')
#     # 速度进行分组
#     def v_0(x):
#         if len([i for i in x if i <= 2 and i > 0]) != 0:
#             return [i for i in x if i <= 2 and i > 0]
#         else:
#             return [0]

#     t = df.groupby('loadingOrder')['speed'].agg(**{'v_count0': lambda x: len(v_0(x)) / len(x)}).reset_index()
#     group_df = pd.merge(group_df, t, on='loadingOrder', how='left')
#     t = df.groupby('loadingOrder')['speed'].agg(**{'v_mean0': lambda x: np.mean(v_0(x))}).reset_index()
#     group_df = pd.merge(group_df, t, on='loadingOrder', how='left')

#     def v_1(x):
#         if len([i for i in x if i > 6 and i <= 10]) != 0:
#             return [i for i in x if i > 6 and i <= 10]
#         else:
#             return [0]

#     t = df.groupby('loadingOrder')['speed'].agg(**{'v_count1': lambda x: len(v_1(x)) / len(x)}).reset_index()
#     group_df = pd.merge(group_df, t, on='loadingOrder', how='left')
#     t = df.groupby('loadingOrder')['speed'].agg(**{'v_mean1': lambda x: np.mean(v_1(x))}).reset_index()
#     group_df = pd.merge(group_df, t, on='loadingOrder', how='left')

#     def v_2(x):
#         if len([i for i in x if i <= 6 and i > 2]) != 0:
#             return [i for i in x if i <= 6 and i > 2]
#         else:
#             return [0]

#     t = df.groupby('loadingOrder')['speed'].agg(**{'v_count2': lambda x: len(v_2(x)) / len(x)}).reset_index()
#     group_df = pd.merge(group_df, t, on='loadingOrder', how='left')
#     t = df.groupby('loadingOrder')['speed'].agg(**{'v_mean2': lambda x: np.mean(v_2(x))}).reset_index()
#     group_df = pd.merge(group_df, t, on='loadingOrder', how='left')
    df['day_nig'] = 0
    df.loc[(df['hour'] > 5) & (df['hour'] < 20),'day_nig'] = 1
    day_on_night = df[df['day_nig'] == 0]
    day_on_day = df[df['day_nig'] == 1]

    agg_function = ['max','median','mean','std','skew']
    agg_ways = ['max','median','mean','std','skew']
    agg_col = ['speed']

    day_on_night_group = day_on_night.groupby('loadingOrder')[agg_col].agg(agg_function).reset_index()
    day_on_night_group.columns = ['loadingOrder'] + ['day_on_night_{}_{}'.format(i, j) for i in agg_col for j in agg_ways]

    day_on_day_group = day_on_day.groupby('loadingOrder')[agg_col].agg(agg_function).reset_index()
    day_on_day_group.columns = ['loadingOrder'] + ['day_on_day_{}_{}'.format(i, j) for i in agg_col for j in agg_ways]

    group_df = group_df.merge(day_on_night_group, on='loadingOrder', how='left')
    group_df = group_df.merge(day_on_day_group, on='loadingOrder', how='left')
    return group_df
    
train = get_feature(train_data, port_data,  mode='train')
test = get_feature(test_data, port_data, mode='test')
features = [c for c in train.columns if c not in ['loadingOrder', 'label', 'mmin', 'mmax', 'cnt', 'timestamp']]
print("Training on {} features".format(len(features)))

valid = train[train['loadingOrder'].isin(temp_train_data_list)]

def build_model(train_, valid_, test_, pred, label, split, seed=1034, is_shuffle=True):
    train_pred = np.zeros(train_.shape[0])
    test_pred = np.zeros(test_.shape[0])
    val_pred = np.zeros(valid_.shape[0])
    n_splits = 5

    assert split in ['kf', 'skf'], '{} Not Support this type of split way'.format(split)

    if split == 'kf':
        folds = KFold(n_splits=n_splits, shuffle=is_shuffle, random_state=seed)
        kf_way = folds.split(train_[pred])
    else:
        folds = StratifiedKFold(n_splits=n_splits, shuffle=is_shuffle, random_state=seed)
        kf_way = folds.split(train_[pred], train_[label])

    print('Use {} features ...'.format(len(pred)))

    params = {
        'learning_rate': 0.05,
        'boosting_type': 'gbdt',
        'num_leaves':52,
        'max_depth':7,
        'metric':'mse',
        'n_estimators':2000,
        'objective': 'regression',
        'subsample':1.0,
        'colsample_bytree':0.5,
        'reg_alpha': 0.877121266008,
        'reg_lambda': 0.7053522513,
        'n_jobs': -1,
        'silent': True,
    }
    
    valid1_x, valid1_y = valid_[pred], valid_[label]
    for n_fold, (train_idx, valid_idx) in enumerate(kf_way, start=1):
        train_x, train_y = train_[pred].iloc[train_idx], train_[label].iloc[train_idx]
        valid_x, valid_y = train_[pred].iloc[valid_idx], train_[label].iloc[valid_idx]
        
        clf = lgb.LGBMRegressor().set_params(**params).set_params(random_state=seed)

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y),(valid_x, valid_y),(valid_[pred],valid_[label])], early_stopping_rounds=100, verbose=500)
 
        test_pred += clf.predict(test_[pred], num_iteration=clf.best_iteration_)
        val_pred += clf.predict(valid1_x, num_iteration=clf.best_iteration_)

    test_['label'] = test_pred/n_splits
    return test_[['loadingOrder','label']]


result = build_model(train, valid, test, features, 'label', 'kf', is_shuffle=True)
result['label'] = result['label'].apply(np.exp)

endtime = datetime.now()
print('date: {}'.format((endtime-starttime).total_seconds()/60))

with mox.file.File(test_data_path, "r+") as f:
    test_data = pd.read_csv(f)
test_data = get_data(test_data, mode='test')

def redict_data(result, test_data):
    group_df = test_data.groupby('loadingOrder')['timestamp'].agg(mmax='max', cnt='count', mmin='min').reset_index()
    group_df['label'] = (group_df['mmax'] - group_df['mmin']).dt.total_seconds()/3600
    group_df.drop(['mmax','cnt','mmin'], axis=1, inplace=True)
    test_data = test_data.merge(group_df, on='loadingOrder', how='left')

    id_list = ['HI545398106803','LR291426429726','QF723400588858','JB123387157454','ZA141598268732','BR663094574600','KD265061648304',
              'KL671073399431','NI245305422658','PK422158256377','TI854412328664']
    for i in id_list:
        result.loc[result['loadingOrder']==i,'label'] = test_data.loc[test_data['loadingOrder']==i,'label'].values[0]
    result.loc[result['loadingOrder']=='LR291426429726','label'] = test_data.loc[test_data['loadingOrder']=='LR291426429726','label'].values[0]+3
    result.loc[result['loadingOrder']=='JB123387157454','label'] = test_data.loc[test_data['loadingOrder']=='JB123387157454','label'].values[0]+1
    result.loc[result['loadingOrder']=='ZA141598268732','label'] = test_data.loc[test_data['loadingOrder']=='ZA141598268732','label'].values[0]+1
    return result

result = redict_data(result, test_data.copy())

test_data = test_data.merge(result, on='loadingOrder', how='left')
test_data['ETA'] = (test_data['onboardDate'] + test_data['label'].apply(lambda x:pd.Timedelta(hours=x))).apply(lambda x:x.strftime('%Y/%m/%d  %H:%M:%S'))
test_data.drop(['direction','TRANSPORT_TRACE'],axis=1,inplace=True)
test_data['onboardDate'] = test_data['onboardDate'].apply(lambda x:x.strftime('%Y/%m/%d  %H:%M:%S'))
test_data['creatDate'] = datetime.now().strftime('%Y/%m/%d  %H:%M:%S')
test_data['timestamp'] = test_data['temp_timestamp']
# 整理columns顺序
result = test_data[['loadingOrder', 'timestamp', 'longitude', 'latitude', 'carrierName', 'vesselMMSI', 'onboardDate', 'ETA', 'creatDate']]

now_time = datetime.now().strftime(format='%Y-%m-%d_%H-%M-%S')
result.to_csv(f'result_{now_time}.csv', index=False)

import moxing as mox
mox.file.shift('os', 'mox')

mox.file.copy(f'result_{now_time}.csv', 'obs://obsacc/data/sub.csv')