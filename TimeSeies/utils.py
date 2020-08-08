# mse：18500

from geopy.distance import distance

# 1. compute distance according to lat_lon (km)

# 1.1 Compute diff distance in two different lat lon by geopy
def diff_distance(x):
    s = [np.nan]
    for i in range(1, len(x)):
        lon1, lat1 = x.iloc[i-1]['longitude'], x.iloc[i-1]['latitude']
        lon2, lat2 = x.iloc[i]['longitude'], x.iloc[i]['latitude']
        distan = distance((lat1,lon1), (lat2,lon2)).km
        s.append(distan)
    return pd.DataFrame({"new":s})
    
# 1.2 Haversine distance
# Reference:<https://ictar.xyz/2015/11/22/[%E8%AF%91]%E6%AF%94%E4%B8%80%E6%AF%94%EF%BC%9APython%E7%9A%84%E4%B8%83%E4%B8%AA%E6%95%B0%E6%8D%AE%E5%8F%AF%E8%A7%86%E5%8C%96%E5%B7%A5%E5%85%B7/>
import math
def calc_dist(row):

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
        source = airports[airports["id"] == row["source_id"]].iloc[0]
        dest = airports[airports["id"] == row["dest_id"]].iloc[0]
        # 使用坐标计算距离
        dist = haversine(dest["longitude"], dest["latitude"], source["longitude"], source["latitude"])
    except (ValueError, IndexError):
        pass
    return dist
    
# 2. Cut DataFrame according to col
# the files are in "dataset/"
def cut_id(x):
    ids = x['loadingOrder'].values[0]
    x.to_csv('dataset/'+str(ids)+'.csv', index=False)
    
data.groupby('loadingOrder')[data.columns].apply(cut_id)
print("Cut compete")