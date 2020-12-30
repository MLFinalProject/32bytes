import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import date

def smooth_target_encoding(data_frame, column_name, target, weight):
    global_mean = data_frame[target].mean()

    agg = data_frame.groupby(column_name)[target].agg(['count', 'mean'])
    sector = data_frame.groupby(column_name)
    categorial_count = agg['count']
    categorial_mean = agg['mean']
    smooth = (categorial_count * categorial_mean + weight * global_mean) / (categorial_count + weight)
    return data_frame[column_name].map(smooth)

month_str2num = {'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6,
                 'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12}
default_values = {'children':0,'country':'NUL','agent':0,'company':0}
# categorial_labels = ['hotel', 'arrival_date_year', 'arrival_date_month', 'arrival_date_week_number', 'arrival_date_day_of_month',
# 					 'meal', 'country', 'market_segment', 'distribution_channel', 'reserved_room_type', 'assigned_room_type',
# 					 'deposit_type', 'agent', 'company', 'customer_type']

# categorial_labels = ['assigned_room_type']

train_df = pd.read_csv('./data/train.csv')
adr_np = train_df.loc[:,'adr'].to_numpy()
train_df = train_df.drop(['ID','is_canceled','reservation_status','reservation_status_date'],axis=1).fillna(value=default_values)

# arrival_date = train_df.loc[:,['arrival_date_year','arrival_date_month','arrival_date_day_of_month']].to_numpy()
# arrival_date[:,1] = [month_str2num[i] for i in arrival_date[:,1]]
# arrival_date = np.array([date(i[0],i[1],i[2]).isoformat() for i in arrival_date])
# arrival_date_id = np.unique(arrival_date, return_inverse = True)[1]

train_df['agent'] = train_df['agent'].apply(str)
train_df['company'] = train_df['company'].apply(str)
train_df['arrival_date_year'] = train_df['arrival_date_year'].apply(str)
train_df['arrival_date_day_of_month'] = train_df['arrival_date_day_of_month'].apply(str)
for label in categorial_labels:
	train_df[label+'_t'] = smooth_target_encoding(train_df, label, 'adr', 300)
train_df = train_df.drop('adr',axis=1)
train_np = pd.get_dummies(train_df).to_numpy()
print(train_np.shape)
# train_np = np.c_[train_np,np.random.rand(len(train_np),1000)]
print(train_np.shape)

reg = LinearRegression().fit(train_np, adr_np)
print(reg.score(train_np, adr_np))