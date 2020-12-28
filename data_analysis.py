import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
month_str2num = {'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6,
                 'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12}

train_df = pd.read_csv('./data/train.csv')
train_label_np = pd.read_csv('./data/train_label.csv').to_numpy()
arrival_date = train_df.loc[:,['arrival_date_year','arrival_date_month','arrival_date_day_of_month']].to_numpy()
arrival_date[:,1] = [month_str2num[i] for i in arrival_date[:,1]]
arrival_date = [date(i[0],i[1],i[2]).isoformat() for i in arrival_date]

number_of_days = train_df.loc[:,['stays_in_weekend_nights', 'stays_in_week_nights']].to_numpy()
number_of_days = np.sum(number_of_days, axis=1)

is_canceled = train_df.loc[:,'is_canceled'].to_numpy()
is_canceled = is_canceled*-1 + 1
adr = train_df.loc[:,'adr'].to_numpy()
total_adr = {}
for i in range(len(is_canceled)):
	try:
		total_adr[arrival_date[i]] += is_canceled[i]*adr[i]*number_of_days[i]	
	except:
		total_adr[arrival_date[i]] = is_canceled[i]*adr[i]*number_of_days[i]

x = []
y = []
for i in range(len(train_label_np)):
	x.append(train_label_np[i][1])
	y.append(total_adr[train_label_np[i][0]])
plt.scatter(x,y,s = 0.1)
plt.show()

