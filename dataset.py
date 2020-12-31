import pandas as pd
import numpy as np
from datetime import date


class Dataset(object):
    """docstring for Dataset"""

    month_str2num = {'January': 1, 'February': 2, 'March': 3,
                     'April': 4, 'May': 5, 'June': 6,
                     'July': 7, 'August': 8, 'September': 9,
                     'October': 10, 'November': 11, 'December': 12}

    def __init__(self):
        self.train_df = pd.read_csv('./data/train.csv')
        self.train_df = self.train_df.drop(self.train_df[(self.train_df.adults+self.train_df.babies+self.train_df.children)==0].index)
        self.train_df['country'].fillna(self.train_df['country'].mode().to_string(), inplace=True)
        self.train_df['children'].fillna(round(self.train_df['children'].mean()), inplace=True)
        self.train_df[['agent','company']] = self.train_df[['agent','company']].fillna(0.0)
        self.train_df[['children', 'company', 'agent']] = self.train_df[['children', 'company', 'agent']].astype('int64')
        self.target_df = self.train_df.loc[:, ['is_canceled', 'adr']]

        
        self.train_df = self.train_df.drop(['ID', 'is_canceled', 'adr',
                                            'reservation_status',
                                            'reservation_status_date'], axis=1)

        arrival_np = self.train_df.loc[:,['arrival_date_year','arrival_date_month','arrival_date_day_of_month']].to_numpy()
        arrival_np[:,1] = [self.month_str2num[i] for i in arrival_np[:,1]]
        self.arrival_df = pd.DataFrame([date(i[0],i[1],i[2]).isoformat() for i in arrival_np], columns = ['arrival_date'])
        self.number_of_days_df = pd.DataFrame(self.train_df['stays_in_weekend_nights'] + 
                                              self.train_df['stays_in_week_nights'], columns = ['number_of_days'])
        self.train_df['agent'] = self.train_df['agent'].apply(str)
        self.train_df['company'] = self.train_df['company'].apply(str)
        self.train_df['arrival_date_year'] = self.train_df['arrival_date_year'].apply(str)
        self.train_df['arrival_date_day_of_month'] = self.train_df['arrival_date_day_of_month'].apply(str)

    def smooth_target_encoding(self, column_name, target, weight = 300, replace = True):
        global_mean = self.target_df[target].mean()
        temp_df = pd.concat([self.train_df[column_name], self.target_df[target]], axis=1)
        agg = temp_df.groupby(column_name)[target].agg(['count', 'mean'])
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
    def get_arrival_date(self, to_numpy = False):
        if to_numpy:
            return self.arrival_df.to_numpy().squeeze()
        else:
            return self.arrival_df
    def get_number_of_days(self, to_numpy = False):
        if to_numpy:
            return self.number_of_days_df.to_numpy().squeeze()
        else:
            return self.number_of_days_df
    def get_adr(self, to_numpy = False):
        if to_numpy:
            return self.target_df[['adr']].to_numpy().squeeze()
        else:
            return self.target_df[['adr']]
    def get_is_canceled(self, to_numpy = False):
        if to_numpy:
            return self.target_df[['is_canceled']].to_numpy().squeeze()
        else:
            return self.target_df[['is_canceled']]
    def get_column(self, column_name, to_numpy = False):
        if to_numpy:
            return self.train_df[[column_name]].to_numpy().squeeze()
        else:
            return self.train_df[[column_name]]
    def get_train_dataset(self, to_numpy = False):
        if to_numpy:
            return pd.get_dummies(self.train_df).to_numpy()
        else:
            return pd.get_dummies(self.train_df)
    def get_test_dataset(self, to_numpy = False):
        return




