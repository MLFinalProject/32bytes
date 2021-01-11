import pandas as pd
import numpy as np
from dataset import Dataset
from classification import *
from regression import *
from feature_engineering import *
from predict import *


hotel_is_cancel = Dataset()
arrival_date_df = hotel_is_cancel.get_test_arrival_date()
number_of_days_df = hotel_is_cancel.get_test_number_of_days()
room_feature = gen_room_feature(hotel_is_cancel.get_feature(['reserved_room_type', 'assigned_room_type']))
net_canceled_feature = gen_net_canceled_feature(hotel_is_cancel.get_feature(['previous_cancellations', 'previous_bookings_not_canceled']))
hotel_is_cancel.add_feature(room_feature)
hotel_is_cancel.add_feature(net_canceled_feature)

remove_only_list = ['country', 'agent']
for only_attribute in remove_only_list:
    attribute_train_column = hotel_is_cancel.get_train_column(only_attribute)
    attribute_test_column = hotel_is_cancel.get_test_column(only_attribute)
    new_attribute_column = remove_only(hotel_is_cancel.get_feature([only_attribute]), attribute_train_column, attribute_test_column)
    hotel_is_cancel.remove_feature([only_attribute])
    hotel_is_cancel.add_feature(new_attribute_column)

hotel_is_cancel.remove_feature(['company'])
hotel_is_cancel.remove_feature(['arrival_date_year'])

x_train_is_canceled = hotel_is_cancel.get_train_dataset()
x_test_is_canceled = hotel_is_cancel.get_test_dataset()
y_train_is_canceled = hotel_is_cancel.get_train_is_canceled()
for only_attribute in remove_only_list:
    remove_string = '{}_RMV'.format(only_attribute)
    x_train_is_canceled.drop([remove_string], axis=1, inplace=True)
    x_test_is_canceled.drop([remove_string], axis=1, inplace=True)

clf = TheRandomForest(x_train_is_canceled, y_train_is_canceled, x_test_is_canceled)





hotel_adr = Dataset()

remove_only_list = ['country', 'agent']
for only_attribute in remove_only_list:
    attribute_train_column = hotel_adr.get_train_column(only_attribute)
    attribute_test_column = hotel_adr.get_test_column(only_attribute)
    new_attribute_column = remove_only(hotel_adr.get_feature([only_attribute]), attribute_train_column, attribute_test_column)
    hotel_adr.remove_feature([only_attribute])
    hotel_adr.add_feature(new_attribute_column)

hotel_adr.remove_feature(['arrival_date_year'])
hotel_adr.remove_feature(['company'])

x_train_adr = hotel_adr.get_train_dataset()
x_test_adr = hotel_adr.get_test_dataset()
y_train_adr = hotel_adr.get_train_adr()

for only_attribute in remove_only_list:
    remove_string = '{}_RMV'.format(only_attribute)
    x_train_adr.drop([remove_string], axis=1, inplace=True)
    x_test_adr.drop([remove_string], axis=1, inplace=True)

reg = TheRandomForestRegressor(x_train_adr, y_train_adr, x_test_adr)

predicts = []
for seed in range(0,100):
    print(f'Start seed {seed}:')
    is_canceled_df = clf.ensemble_seed(seed)
    adr_df = reg.ensemble_seed(seed)
    predict_df = predict_ensemble(pd.concat([arrival_date_df, is_canceled_df, adr_df, number_of_days_df], axis=1))
    predicts.append(predict_df['label'])
    print('--------------------')
predicts = np.stack(predicts)
predicts = np.around(np.mean(predicts, axis=0))
output_df = predict_df.copy()
output_df['label'] = predicts
output_df.to_csv('output_100_1.csv')


