import pandas as pd
import numpy as np
from datetime import date


class Dataset(object):
    """docstring for Dataset"""

    month_str2num = {'January': 1, 'February': 2, 'March': 3,
                     'April': 4, 'May': 5, 'June': 6,
                     'July': 7, 'August': 8, 'September': 9,
                     'October': 10, 'November': 11, 'December': 12}

    default_values = {'children': 0, 'country': 'NUL',
                      'agent': 0, 'company': 0}

    def __init__(self):
        self.train_df = pd.read_csv('./data/train.csv')
        self.target_df = self.train_df.loc[:, ['is_canceled', 'adr']]

        self.train_df = self.train_df.fillna(value=self.default_values)
        self.train_df = self.train_df.drop(['ID', 'is_canceled', 'adr',
                                            'reservation_status',
                                            'reservation_status_date'], axis=1)

        arrival_np = self.train_df.loc[:,['arrival_date_year','arrival_date_month','arrival_date_day_of_month']].to_numpy()
        arrival_np[:,1] = [self.month_str2num[i] for i in arrival_np[:,1]]
        self.arrival_df = pd.DataFrame([date(i[0],i[1],i[2]).isoformat() for i in arrival_np], columns = ['arrival_date'])

        self.train_df['agent'] = self.train_df['agent'].apply(str)
        self.train_df['company'] = self.train_df['company'].apply(str)
        self.train_df['arrival_date_year'] = self.train_df['arrival_date_year'].apply(str)
        self.train_df['arrival_date_day_of_month'] = self.train_df['arrival_date_day_of_month'].apply(str)

    def smooth_target_encoding(self, column_name, target, weight = 300, replace = True):
        global_mean = self.target_df[target].mean()
        temp_df = pd.concat([self.train_df[column_name], self.target_df[target]], axis=1)
        agg = temp_df.groupby(column_name)[target].agg(['count', 'mean'])
        sector = temp_df.groupby(column_name)
        categorial_count = agg['count']
        categorial_mean = agg['mean']
        smooth = (categorial_count * categorial_mean + weight * global_mean) / (categorial_count + weight)
        if replace:
            self.train_df[column_name] = temp_df[column_name].map(smooth)
        else:
            self.train_df[column_name + '_target_encoding'] = temp_df[column_name].map(smooth)
        return
    def one_hot_encoding(self, column_name):
        self.train_df[column_name] = pd.get_dummies(self.train_df[column_name], prefix = column_name)
        return
    def get_arrival_date(self):
        return self.arrival_df
    def get_adr(self):
        return self.target_df[['adr']]
    def get_is_canceled(self):
        return self.target_df[['is_canceled']]

categorial_labels = ['hotel', 'arrival_date_year', 'arrival_date_month', 'arrival_date_week_number', 'arrival_date_day_of_month',
                   'meal', 'country', 'market_segment', 'distribution_channel', 'reserved_room_type', 'assigned_room_type',
                   'deposit_type', 'agent', 'company', 'customer_type']
categorial_labels = []


from sklearn.linear_model import LinearRegression
p = Dataset()
input_df = pd.concat([p.get_arrival_date(), p.get_is_canceled(), p.get_adr()], axis=1)
print(input_df)
exit()
for label in categorial_labels:
    p.one_hot_encoding(label, False)
dfnp = pd.get_dummies(p.train_df).to_numpy()
reg = LinearRegression().fit(dfnp, p.target_df['adr'].to_numpy())
print(reg.score(dfnp, p.target_df['adr'].to_numpy()))


