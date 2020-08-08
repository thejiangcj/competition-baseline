#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install --upgrade pip')
get_ipython().system('pip install --upgrade tqdm geopy gensim lightgbm scikit-learn')
get_ipython().system('pip install --upgrade pandas')


# In[45]:


import pandas as pd
from tqdm import tqdm
import numpy as np
from datetime import datetime
import gc

from zipfile import ZipFile
from sklearn.preprocessing import LabelEncoder

import moxing as mox
mox.file.shift('os', 'mox')

# seed = 2020
# np.random.seed(seed)

train_gps_path = 'obs://obsacc/data/train_data_01-1.csv'
test_data_path = 'obs://obsacc/data/data/R2_ATest 0711.csv'
port_data_path = 'obs://obsacc/data/data/port_ed.csv'

with mox.file.File(train_gps_path, "r") as f:
    train_data = pd.read_csv(f)

with mox.file.File(test_data_path, "r") as f:
    test_data = pd.read_csv(f)

with mox.file.File(port_data_path, "r") as f:
    port_data = pd.read_csv(f)

train_data.drop(['vesselStatus', 'vesselDatasource'],axis=1,inplace=True)

#################
data_id = train_data['loadingOrder']
train_data = train_data.groupby(['loadingOrder'],as_index=False).fillna(method='ffill').fillna(method='bfill')
train_data['loadingOrder'] = data_id
trace_is_na = train_data['TRANSPORT_TRACE'].notna()
train_data = train_data[trace_is_na]
#################

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

del data_id, trace_is_na
gc.collect()


# In[46]:


not_list = ['AC860038925693','CS952075060675','DM428031991357','DS626552529494','EI581767201011','GA472803281061','HL358914564422','JE845105704656','LK919030439899',
'LR291426429726',
'LY233998601535',
'NJ417242079579',
'PP710466021916',
'PQ602767500334',
'QF723400588858',
'UK663883669352',
'VJ323567531982',
'ZQ798500357614',
'ZS950908209190']
test_data = test_data[~test_data['loadingOrder'].isin(not_list)]


# In[47]:


# 1. 消除偏移
# 9300订单
tmp_train_data = train_data[['loadingOrder','longitude','latitude']].copy()
tmp_train_data['longitude'] = tmp_train_data['longitude'].apply(np.abs)
temp_diff = tmp_train_data.groupby('loadingOrder')[['longitude','latitude']].diff(1)
temp_diff['loadingOrder'] = tmp_train_data['loadingOrder']

def abs_max(x):
    x = np.abs(x)
    return x.max()

temp_agg = temp_diff.groupby('loadingOrder')[['longitude','latitude']].agg(longitude=('longitude', abs_max), latitude=('latitude', abs_max)).reset_index()
temp_agg = temp_agg[(temp_agg['longitude']<5)&(temp_agg['latitude']<5)]
train_idx = temp_agg['loadingOrder'].unique().tolist()
train_data = train_data[train_data['loadingOrder'].isin(train_idx)]

# 7. 时间偏移
temp_diff = train_data.groupby('loadingOrder')['timestamp'].diff(1).to_frame()
temp_diff['loadingOrder'] = train_data['loadingOrder']
temp_diff['timestamp'] = temp_diff['timestamp'].dt.total_seconds()/3600
temp_agg = temp_diff.groupby('loadingOrder')['timestamp'].agg('max').reset_index()
temp_agg = temp_agg[temp_agg['timestamp']<12]
train_idx = temp_agg['loadingOrder'].unique().tolist()
train_data = train_data[train_data['loadingOrder'].isin(train_idx)]

train_data.sort_values(['loadingOrder','timestamp'], inplace=True)


# In[48]:


train_data.reset_index(drop=True)
trace = train_data.groupby('loadingOrder')['TRANSPORT_TRACE'].agg(max='max').reset_index()
test_trace_list = test_data['TRANSPORT_TRACE'].unique().tolist()
test_trace_id_list = trace[trace['max'].isin(test_trace_list)]['loadingOrder'].unique().tolist()

# 在test的trace中的train数据集
train_data_is_in_test = train_data[train_data['loadingOrder'].isin(test_trace_id_list)]
train_data_is_in_test['is_in_test'] = 0

# 继续挖掘不在其中的
train_data_notin_test = train_data[~train_data['loadingOrder'].isin(test_trace_id_list)]

del train_data, temp_diff, tmp_train_data, temp_agg, train_idx
gc.collect()


# In[49]:


# 在测试集但不在训练集的trace
print("In test_dataset but not in train_dataset:")
for i in train_data_is_in_test['TRANSPORT_TRACE'].unique().tolist():
    if i not in test_trace_list:
        print(i)


# In[50]:


# 添加所有在4-6月份的数据
time = train_data_notin_test.groupby('loadingOrder')['timestamp'].agg(max='max',min='min').reset_index()
time['start_date_year'] = time['min'].dt.year
time['start_date_month'] = time['min'].dt.month
time['end_date_year'] = time['max'].dt.year
time['end_date_month'] = time['max'].dt.month

is_2020 = time['start_date_year'].apply(lambda x:x==2020)
is_start_4 = time['start_date_month'].apply(lambda x:x<4)
is_end_4 = time['end_date_month'].apply(lambda x:x>=4)
is_start_6 = time['start_date_month'].apply(lambda x:x==4|x==5|x==6)

time = time[(is_2020&is_start_4&is_end_4)|(is_2020&is_start_6)]
time_list_id = time['loadingOrder'].unique().tolist()
is_four = train_data_notin_test[train_data_notin_test['loadingOrder'].isin(time_list_id)]
is_four['is_in_test'] = 1

print("1. After split trace: ", train_data_is_in_test['loadingOrder'].nunique())

train_data_is_in_test = train_data_is_in_test.append(is_four)

print("2. After append split trace: ", train_data_is_in_test['loadingOrder'].nunique())

# all_trace是否在重点经纬度内
tmp_trace = train_data_is_in_test.groupby('loadingOrder')['TRANSPORT_TRACE'].agg(max='max').reset_index()
tmp_trace['end_port'] = tmp_trace['max'].str.split('-').apply(lambda x:x[-1])
tmp_trace


# In[51]:


ports = port_data['TRANS_NODE_NAME'].unique().tolist()
tmp_trace[~tmp_trace['end_port'].isin(ports)]


# In[52]:


## 提取port_data
port = pd.DataFrame(columns=port_data.columns)
def extrace_lat(x, port_data):
    global port
    port = port.append(port_data[port_data['TRANS_NODE_NAME']==x])

tmp_trace['end_port'].apply(extrace_lat, args=(port_data,))
port.drop_duplicates(inplace=True)
tmp_trace = tmp_trace.merge(port, left_on='end_port', right_on='TRANS_NODE_NAME', how='left')

tmp_trace.columns = ['loadingOrder','trace','end_port','TRANS_NODE_NAME','end_port_lat','end_port_lon']
tmp_trace.drop(['TRANS_NODE_NAME'], axis=1, inplace=True)

print("3. Start find last: ", train_data_is_in_test['loadingOrder'].nunique())
train_data_is_in_test.sort_values(['loadingOrder','timestamp'], inplace=True)
train_data_is_in_test.reset_index(drop=True)

tmp_data = train_data_is_in_test.groupby('loadingOrder', as_index=False).tail(1).reset_index(drop=True)[['loadingOrder','latitude','longitude']]
tmp_data.columns = ['loadingOrder','last_lat','last_lon']
tmp_trace = tmp_trace.merge(tmp_data, on='loadingOrder', how='left')

# 添加label
group_df = train_data_is_in_test.groupby('loadingOrder')['timestamp'].agg(mmax='max', cnt='count', mmin='min').reset_index()
group_df['label'] = (group_df['mmax'] - group_df['mmin']).dt.total_seconds()/3600
group_df.drop(['mmax','cnt','mmin'], axis=1, inplace=True)
train_data_is_in_test = train_data_is_in_test.merge(group_df, on='loadingOrder', how='left')


# In[53]:


tmp_data = train_data_is_in_test.groupby('loadingOrder').tail(2).reset_index(drop=True)
import math
def compute_distance(row):
    def haversine(lon1, lat1, lon2, lat2):
        # 将坐标转换为浮点数
        lon1, lat1, lon2, lat2 = [float(lon1), float(lat1), float(lon2), float(lat2)]
        # 将度数转换为弧度
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
        # 计算距离
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a)) 
        km = 6367 * c
        return km
 
    dist = 0
    try:
        # 匹配源和目的以获得坐标
        source = row.iloc[0]
        dest = row.iloc[-1]
        # 使用坐标计算距离
        dist = haversine(dest["longitude"], dest["latitude"], source["longitude"], source["latitude"])
    except (ValueError, IndexError):
        pass
    hour = (dest['timestamp'] - source['timestamp']).total_seconds()/3600
    row['handle_speed'] = dist/hour
    return row[['loadingOrder','handle_speed','label']].drop_duplicates(subset=['loadingOrder','handle_speed'])
    
tmp_data = tmp_data.groupby(['loadingOrder']).apply(compute_distance).reset_index(drop=True)
tmp_trace = tmp_trace.merge(tmp_data, on='loadingOrder', how='left')


# In[54]:


tmp_trace


# In[55]:


def compute_hour(x):
    def haversine(lon1, lat1, lon2, lat2):
        # 将坐标转换为浮点数
        lon1, lat1, lon2, lat2 = [float(lon1), float(lat1), float(lon2), float(lat2)]
        # 将度数转换为弧度
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
        # 计算距离
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a)) 
        km = 6367 * c
        return km
 
    dist = 0
    try:
        # 匹配源和目的以获得坐标
        # 使用坐标计算距离
        dist = haversine(x["end_port_lon"], x["end_port_lat"], x["last_lon"], x["last_lat"])
    except (ValueError, IndexError):
        pass
    
    if x['handle_speed'] < 2:
        x['label'] = x['label'] + dist/1
    else:
        x['label'] = x['label'] + dist/((x['handle_speed']+0.001)/2)
    return x[['loadingOrder','label']].reset_index(drop=True)

tmp_data = tmp_trace.apply(compute_hour, axis=1).reset_index(drop=True)
tmp_data.columns = ['loadingOrder','label']
train_data_is_in_test = train_data_is_in_test.merge(tmp_data, on='loadingOrder', how='left')
#train_data_is_in_test.drop(['label'], axis=1, inplace=True)

# 过滤太大的label
print("start filter label")
train_data_is_in_test = train_data_is_in_test[(train_data_is_in_test['label_y'] - train_data_is_in_test['label_x'])<500]

train_data_is_in_test.drop(['label_x'], axis=1, inplace=True)
train_data_is_in_test.rename({'label_y':'label'}, axis=1, inplace=True)

print("3. After add label: ", train_data_is_in_test['loadingOrder'].nunique())

s = train_data_is_in_test.groupby('loadingOrder')['timestamp'].agg(cnt='count').reset_index()
s_list = s[s['cnt'].apply(lambda x:True if x>30 else False)]['loadingOrder'].unique().tolist()
train_data_is_in_test = train_data_is_in_test[train_data_is_in_test['loadingOrder'].isin(s_list)]
train_data_is_in_test


# In[56]:


train_data_is_in_test.groupby(['loadingOrder'])['label'].agg(label='mean').reset_index()


# In[58]:


train_data_is_in_test.to_csv('train_05_1.csv', index=False)
import moxing as mox
mox.file.shift('os', 'mox')
mox.file.copy('./train_05_1.csv', 'obs://obsacc/data/train_05_1.csv')