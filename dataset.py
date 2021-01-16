import pandas as pd
import numpy as np
from datetime import date

class Dataset(object):
    """docstring for Dataset"""

    month_str2num = {'January': 1, 'February': 2, 'March': 3,
                     'April': 4, 'May': 5, 'June': 6,
                     'July': 7, 'August': 8, 'September': 9,
                     'October': 10, 'November': 11, 'December': 12}
    data_label = {'train': 0, 'test': 1}
    
    def __init__(self):
        train_df = pd.read_csv('./data/train.csv')
        test_df = pd.read_csv('./data/test.csv')
        self.target_df = train_df.loc[:, ['is_canceled', 'adr']]
        train_df.drop(['adr', 'is_canceled','reservation_status','reservation_status_date'], axis = 1, inplace = True)
        self.train_test_df = pd.concat([train_df.assign(dataset=self.data_label["train"]), test_df.assign(dataset=self.data_label["test"])], ignore_index = True)
        self.train_test_df.drop(['ID'], axis = 1, inplace = True)
        arrival_np = self.train_test_df.loc[:,['arrival_date_year','arrival_date_month','arrival_date_day_of_month']].to_numpy()
        arrival_np[:,1] = [self.month_str2num[i] for i in arrival_np[:,1]]
        self.arrival_date_df = pd.DataFrame([date(i[0],i[1],i[2]).isoformat() for i in arrival_np], columns = ['arrival_date'])
        self.arrival_date_df['dataset'] = self.train_test_df['dataset']
        self.number_of_days_df = pd.DataFrame(self.train_test_df['stays_in_weekend_nights'] + 
                                    self.train_test_df['stays_in_week_nights'], columns = ['number_of_days'])
        self.number_of_days_df['dataset'] = self.train_test_df['dataset']

        self.train_test_df['country'].fillna(self.train_test_df['country'].mode().to_string(), inplace=True)
        self.train_test_df['children'].fillna(round(self.train_test_df['children'].mean()), inplace=True)
        self.train_test_df[['agent','company']] = self.train_test_df[['agent','company']].fillna(0.0)
        self.train_test_df[['children', 'company', 'agent']] = self.train_test_df[['children', 'company', 'agent']].astype('int64')
        
        self.train_test_df['agent'] = self.train_test_df['agent'].apply(str)
        self.train_test_df['company'] = self.train_test_df['company'].apply(str)
        self.train_test_df['arrival_date_year'] = self.train_test_df['arrival_date_year'].apply(str)
        self.train_test_df['arrival_date_day_of_month'] = self.train_test_df['arrival_date_day_of_month'].apply(str)
        
    def smooth_target_encoding(self, column_name, target, weight = 300, replace = True):
        train_df = self.train_test_df[self.train_test_df['dataset'].eq(self.data_label["train"])].drop(['dataset'], axis = 1)
        global_mean = self.target_df[target].mean()
        temp_df = pd.concat([train_df[column_name], self.target_df[target]], axis=1)
        agg = temp_df.groupby(column_name)[target].agg(['count', 'mean'])
        categorial_count = agg['count']
        categorial_mean = agg['mean']
        smooth = (categorial_count * categorial_mean + weight * global_mean) / (categorial_count + weight)
        if replace:
            self.train_test_df[column_name] = temp_df[column_name].map(smooth)
        else:
            self.train_test_df[column_name + '_target_encoding'] = temp_df[column_name].map(smooth)
        return
    def one_hot_encoding(self, column_name, replace = True):
        self.train_test_df = pd.concat([self.train_test_df, pd.get_dummies(self.train_test_df[column_name], prefix = column_name)],axis = 1)
        if replace:
            self.train_test_df.drop([column_name], axis = 1, inplace = True)
        return


    def add_feature(self, df):
        self.train_test_df = pd.concat([self.train_test_df, df], axis=1)
        
    def get_feature(self, column_name_list):
        return self.train_test_df[column_name_list]

    def remove_feature(self, column_name_list):
        self.train_test_df.drop(column_name_list, axis = 1, inplace = True)



    def get_train_dataset(self):
        train_test_df = pd.get_dummies(self.train_test_df)
        train_df = train_test_df[train_test_df['dataset'].eq(self.data_label["train"])].drop(['dataset'], axis = 1)
        return train_df

    def get_train_arrival_date(self):
        temp_df = self.arrival_date_df[self.arrival_date_df['dataset'].eq(self.data_label["train"])].drop(['dataset'], axis = 1)
        return temp_df

    def get_train_number_of_days(self):
        temp_df = self.number_of_days_df[self.number_of_days_df['dataset'].eq(self.data_label["train"])].drop(['dataset'], axis = 1)
        return temp_df

    def get_train_column(self, column_name):
        temp_df = self.train_test_df[self.train_test_df['dataset'].eq(self.data_label["train"])]
        return temp_df[[column_name]]

    def get_train_adr(self):
        return self.target_df[['adr']]

    def get_train_is_canceled(self):
        return self.target_df[['is_canceled']]
    
    
    def get_test_dataset(self):
        train_test_df = pd.get_dummies(self.train_test_df)
        test_df = train_test_df[train_test_df['dataset'].eq(self.data_label["test"])].drop(['dataset'], axis = 1)
        return test_df.reset_index(drop = True)

    def get_test_arrival_date(self):
        temp_df = self.arrival_date_df[self.arrival_date_df['dataset'].eq(self.data_label["test"])].drop(['dataset'], axis = 1)
        return temp_df.reset_index(drop = True)

    def get_test_number_of_days(self):
        temp_df = self.number_of_days_df[self.number_of_days_df['dataset'].eq(self.data_label["test"])].drop(['dataset'], axis = 1)
        return temp_df.reset_index(drop = True)

    def get_test_column(self, column_name):
        temp_df = self.train_test_df[self.train_test_df['dataset'].eq(self.data_label["test"])]
        return temp_df[[column_name]]


