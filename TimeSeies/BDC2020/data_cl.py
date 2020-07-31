import pandas as pd
from tqdm import tqdm
import numpy as np
from datetime import datetime
import gc

from zipfile import ZipFile
from sklearn.preprocessing import LabelEncoder

import moxing as mox
mox.file.shift('os', 'mox')

seed = 2020
np.random.seed(seed)


train_gps_path=ZipFile('train0711.zip')

test_data_path = 'obs://obsacc/data/data/R2_ATest 0711.csv'
port_data_path = 'obs://obsacc/data/data/port.csv'

names = ['loadingOrder','carrierName','timestamp','longitude',
                  'latitude','vesselMMSI','speed','direction','vesselNextport',
                  'vesselNextportETA','vesselStatus','vesselDatasource','TRANSPORT_TRACE']

train_data = pd.DataFrame(columns=names)
train_data.drop(['vesselNextport','vesselNextportETA'],axis=1,inplace=True)

col_list = []

with train_gps_path.open('train0711.csv') as f:
    all_data = pd.read_csv(f, header=None, names=names, chunksize=10000000)

    for data in tqdm(all_data):
        data.drop(['vesselNextport','vesselNextportETA'],axis=1,inplace=True)
        
        data.dropna(thresh=4, inplace=True)
        data['latitude'] = data['latitude'].round(4)
        data['longitude'] = data['longitude'].round(4)
        data.drop_duplicates(subset=['loadingOrder','longitude','latitude','speed'],inplace=True)
        data.drop_duplicates(subset=['loadingOrder','timestamp','carrierName','vesselMMSI'],inplace=True)
        train_data = train_data.append(data)

    del all_data
    gc.collect()
    
with mox.file.File(test_data_path, "r") as f:
    test_data = pd.read_csv(f)

with mox.file.File(port_data_path, "r") as f:
    port_data = pd.read_csv(f)
    
train_data.drop_duplicates(subset=['loadingOrder','longitude','latitude','speed'],inplace=True)
train_data.drop_duplicates(subset=['loadingOrder','timestamp','carrierName','vesselMMSI'],inplace=True)

# 1. 排序
train_data.sort_values(['loadingOrder','timestamp'], inplace=True)

train_data.to_csv('train_data_01-1.csv', index=False)
mox.file.copy('./train_data_01-1.csv', 'obs://obsacc/data/train_data_01-1.csv')

# 2. Convert values
# train_data
import pandas as pd
from tqdm import tqdm
import numpy as np
from datetime import datetime
import gc

from zipfile import ZipFile
from sklearn.preprocessing import LabelEncoder

import moxing as mox
mox.file.shift('os', 'mox')

seed = 2020
np.random.seed(seed)

train_gps_path = 'obs://obsacc/data/train_data_01-1.csv'
test_data_path = 'obs://obsacc/data/data/R2_ATest 0711.csv'
port_data_path = 'obs://obsacc/data/data/port.csv'

with mox.file.File(train_gps_path, "r") as f:
    train_data = pd.read_csv(f)

with mox.file.File(test_data_path, "r") as f:
    test_data = pd.read_csv(f)

with mox.file.File(port_data_path, "r") as f:
    port_data = pd.read_csv(f)


train_data['timestamp'] = pd.to_datetime(train_data['timestamp'], infer_datetime_format=True)
train_data['speed'] = train_data['speed'].astype(int)
train_data['direction'] = train_data['direction'].astype(int)
train_data['TRANSPORT_TRACE'] = train_data['TRANSPORT_TRACE'].astype(str)
# test_data
test_data['TRANSPORT_TRACE'] = test_data['TRANSPORT_TRACE'].astype(str)
# port_data
port_data = port_data[['TRANS_NODE_NAME','LATITUDE','LONGITUDE']]
port_data['TRANS_NODE_NAME'] = port_data['TRANS_NODE_NAME'].astype(str)

port_data.drop_duplicates(subset=['TRANS_NODE_NAME'],inplace=True)

## drop not trace
trace = train_data.groupby('loadingOrder')['TRANSPORT_TRACE'].agg(max='max').reset_index()
trace_list = trace[trace['max'].notnull()]['loadingOrder'].unique().tolist()
train_data = train_data[train_data['loadingOrder'].isin(trace_list)]
train_data = train_data[train_data['TRANSPORT_TRACE'].str.contains('-', regex=False)]

test_data['TRANSPORT_TRACE'] = test_data['TRANSPORT_TRACE'].str.split('-')
test_data['begin_path'] = test_data['TRANSPORT_TRACE'].apply(lambda x:x[0])
test_data['end_path'] = test_data['TRANSPORT_TRACE'].apply(lambda x:x[-1])
test_begin_unique = test_data['begin_path'].unique().tolist()
test_end_unique = test_data['end_path'].unique().tolist()
test_data_list = test_begin_unique+test_end_unique
#test_data_list = list(set(test_data_list))

temp_train_data = train_data.groupby('loadingOrder')['TRANSPORT_TRACE'].agg(trace='max').reset_index()
temp_train_data['trace'] = temp_train_data['trace'].str.split('-')
temp_is = temp_train_data['trace'].apply(lambda x:True if len(x)>=2 else False)
temp_train_data = temp_train_data[temp_is] #3619
print(temp_train_data['loadingOrder'].nunique())


def filter_data(x, start_list, end_list):
#     if len(x)>=2:
#         return True
#     return False
    # 测试集开始和结束出现港口的列表
    # x为list类型, 比如[CNSHK, ZADUR], [CNSHK,MXZLO]等
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
train_data = train_data[train_data['loadingOrder'].isin(temp_train_data_list)]
print(train_data['loadingOrder'].nunique())
#######

# 验证出发和到达港是否正确
def extract_latlon(x, port_data):
    lat,lon = port_data[port_data['TRANS_NODE_NAME']==x]['LATITUDE'].values[0], port_data[port_data['TRANS_NODE_NAME']==x]['LONGITUDE'].values[0]
    return (lat,lon)

tmp_trace = train_data.groupby('loadingOrder')['TRANSPORT_TRACE'].agg(trace='max').reset_index()
tmp_trace['trace'] = tmp_trace['trace'].str.split('-')
tmp_trace['start_port'] = tmp_trace['trace'].apply(lambda x:x[0])
tmp_trace['end_port'] = tmp_trace['trace'].apply(lambda x:x[-1])

port_data_list = port_data['TRANS_NODE_NAME'].unique().tolist()
tmp_trace = tmp_trace[tmp_trace['end_port'].isin(port_data_list)&tmp_trace['start_port'].isin(port_data_list)]

tmp_trace['start_port_latlon'] = tmp_trace['start_port'].apply(extract_latlon, args=(port_data,))
tmp_trace['end_port_latlon'] = tmp_trace['end_port'].apply(extract_latlon, args=(port_data,))
del port_data_list
    # 提取出发和回来经纬度

def conpute_is_in_max(x, df):
    lat,lon = df.iloc[x['idxmax']]['latitude'], df.iloc[x['idxmax']]['longitude']
    return (lat,lon)

def conpute_is_in_min(x, df):
    lat,lon = df.iloc[x['idxmin']]['latitude'], df.iloc[x['idxmin']]['longitude']
    return (lat,lon)

train_data = train_data.reset_index(drop=True)
is_in_start_end = train_data.groupby('loadingOrder')['timestamp'].agg(idxmax='idxmax', idxmin='idxmin').reset_index()
is_in_start_end['gps_end_port'] = is_in_start_end[['idxmax','idxmin']].apply(conpute_is_in_max, args=(train_data,), axis=1)
is_in_start_end['gps_start_port'] = is_in_start_end[['idxmax','idxmin']].apply(conpute_is_in_min, args=(train_data,), axis=1)
is_in_start_end.drop(['idxmax','idxmin'], axis=1, inplace=True)
print(train_data['loadingOrder'].nunique())
# 合并
tmp_trace = tmp_trace.merge(is_in_start_end, on='loadingOrder', how='left')

def isin_dim_start(x):
    dis_lat = abs(x['start_port_latlon'][0] - x['gps_start_port'][0])
    dis_lon = abs(x['start_port_latlon'][-1] - x['gps_start_port'][-1])

    if dis_lat<2 and dis_lon<2:
        return True
    else:
        return False

def isin_dim_end(x):
    dis_lat = abs(x['end_port_latlon'][0] - x['gps_end_port'][0])
    dis_lon = abs(x['end_port_latlon'][-1] - x['gps_end_port'][-1])

    if dis_lat<2 and dis_lon<2:
        return True
    else:
        return False

is_start = tmp_trace[['start_port_latlon','gps_start_port']].apply(isin_dim_start, axis=1)
is_end = tmp_trace[['end_port_latlon','gps_end_port']].apply(isin_dim_end, axis=1)

tmp_trace = tmp_trace[is_start&is_end]
idlist = tmp_trace['loadingOrder'].unique().tolist()

train_data = train_data[train_data['loadingOrder'].isin(idlist)]
print(train_data['loadingOrder'].nunique())

del tmp_trace
del port_data
del is_in_start_end
del is_start
del is_end
del idlist
gc.collect()
#######

# 3. 填充空置
train_data = train_data.replace([np.inf, -np.inf], np.nan).dropna(subset=["speed","direction"], how="any")
print("train_data NULL value is:")
print(train_data.isna().sum())
train_data_id = train_data['loadingOrder'].copy()
train_data = train_data.groupby('loadingOrder').fillna(method="bfill").fillna(method="ffill")
train_data['loadingOrder'] = train_data_id.copy()

del train_data_id
gc.collect()

# 4. label_encode
print("4. Start LabelEncoder")
encode_label = ['carrierName', 'vesselMMSI', 'vesselStatus', 'vesselDatasource']
train_data[encode_label] = train_data[encode_label].apply(LabelEncoder().fit_transform)

# 5. final
# 添加label
group_df = train_data.groupby('loadingOrder')['timestamp'].agg(mmax='max', cnt='count', mmin='min').reset_index()
group_df['label'] = (group_df['mmax'] - group_df['mmin']).dt.total_seconds()/3600
group_df.drop(['mmax','cnt','mmin'], axis=1, inplace=True)
train_data = train_data.merge(group_df, on='loadingOrder', how='left')

# 6. 消除偏移
tmp_train_data = train_data[['loadingOrder','longitude','latitude']].copy()
tmp_train_data['longitude'] = tmp_train_data['longitude'].apply(np.abs)
temp_diff = tmp_train_data.groupby('loadingOrder')[['longitude','latitude']].diff(1)
temp_diff['loadingOrder'] = tmp_train_data['loadingOrder']

def abs_max(x):
    x = np.abs(x)
    return x.max()

temp_agg = temp_diff.groupby('loadingOrder')[['longitude','latitude']].agg(longitude=('longitude', abs_max), latitude=('latitude', abs_max)).reset_index()
temp_agg = temp_agg[(temp_agg['longitude']<3)&(temp_agg['latitude']<3)]
train_idx = temp_agg['loadingOrder'].unique().tolist()
train_data = train_data[train_data['loadingOrder'].isin(train_idx)]
print(train_data['loadingOrder'].nunique())

# 7. 时间偏移
temp_diff = train_data.groupby('loadingOrder')['timestamp'].diff(1).to_frame()
temp_diff['loadingOrder'] = train_data['loadingOrder']
temp_diff['timestamp'] = temp_diff['timestamp'].dt.total_seconds()/3600
temp_agg = temp_diff.groupby('loadingOrder')['timestamp'].agg('max').reset_index()
temp_agg = temp_agg[temp_agg['timestamp']<12]
train_idx = temp_agg['loadingOrder'].unique().tolist()
train_data = train_data[train_data['loadingOrder'].isin(train_idx)]

del temp_diff
del temp_agg
gc.collect()

train_data.sort_values(['loadingOrder','timestamp'], inplace=True)

s = train_data.groupby('loadingOrder')['timestamp'].agg(cnt='count').reset_index()
s_list = s[s['cnt'].apply(lambda x:True if x>30 else False)]['loadingOrder'].unique().tolist()
train_data = train_data[train_data['loadingOrder'].isin(s_list)]

# 8. sample
tmp_train = train_data.groupby('loadingOrder').apply(lambda x: x.sample(n=30,replace=False)).reset_index(drop=True)
tmp_train.sort_values(['loadingOrder','timestamp'], inplace=True)

train_data.to_csv('train_data_03.csv', index=False)
mox.file.copy('./train_data_03.csv', 'obs://obsacc/data/train_data_03.csv')