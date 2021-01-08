import pandas as pd

from dataset import Dataset
from classification import *
from regression import *
from feature_engineering import *
from predict import predict
from encoder import *


encode_target = ['hotel','agent','arrival_date_day_of_month','arrival_date_year','assigned_room_type','company','country','customer_type','deposit_type','distribution_channel','market_segment','meal','reserved_room_type']

hotel_is_cancel = Dataset()

# removing outliers
# attribute_threshold_dict = {"adults":3, "babies":2, "children":3, "required_car_parking_spaces":2, "stays_in_week_nights":20, "stays_in_weekend_nights":10}
# for key in attribute_threshold_dict:
#     new_attribute_df = transfer_not_enough_data_to_mean(hotel_is_cancel.get_feature([key]), attribute_threshold_dict[key])
#     hotel_is_cancel.remove_feature([key])
#     hotel_is_cancel.add_feature(new_attribute_df)

# modified_key = "arrival_date_week_number"
# peak = 34
# new_attribute_df = absolute_peak_transform(hotel_is_cancel.get_feature([modified_key]), peak)
# hotel_is_cancel.remove_feature([modified_key])
# hotel_is_cancel.add_feature(new_attribute_df)


room_feature = gen_room_feature(hotel_is_cancel.get_feature(['reserved_room_type', 'assigned_room_type']))
net_canceled_feature = gen_net_canceled_feature(hotel_is_cancel.get_feature(['previous_cancellations', 'previous_bookings_not_canceled']))
hotel_is_cancel.add_feature(room_feature)
hotel_is_cancel.add_feature(net_canceled_feature)

# ---remove only_feature---
# country_column = hotel_is_cancel.get_feature(['country'])
remove_only_list = ['country', 'agent', 'company']
for only_attribute in remove_only_list:
    attribute_train_column = hotel_is_cancel.get_train_column(only_attribute)
    attribute_test_column = hotel_is_cancel.get_test_column(only_attribute)
    new_attribute_column = remove_only(hotel_is_cancel.get_feature([only_attribute]), attribute_train_column, attribute_test_column)
    # print(new_country_column)
    # print(new_attribute_column[only_attribute].value_counts().loc['RMV'])
    hotel_is_cancel.remove_feature([only_attribute])
    hotel_is_cancel.add_feature(new_attribute_column)
# exit()
# -------------------------

# From Hsu
# data_not_encoded = pd.concat([hotel_is_cancel.get_feature(encode_target),hotel_is_cancel.get_train_is_canceled()],axis = 1)
# data_encoded = target_encode(data_not_encoded[encode_target],data_not_encoded['is_canceled'])
# data_encoded_col_name = data_encoded.columns
# for col in data_encoded_col_name:
# 	data_encoded[col].fillna(data_encoded[col].mean(),inplace = True)
# hotel_is_cancel.add_feature(data_encoded)
# hotel_is_cancel.remove_feature(encode_target)

# ---Remove only---
x_train_is_canceled = hotel_is_cancel.get_train_dataset()
x_test_is_canceled = hotel_is_cancel.get_test_dataset()
y_train_is_canceled = hotel_is_cancel.get_train_is_canceled()

# print(type(hotel_is_cancel))
# print(type(x_train_is_canceled))
for only_attribute in remove_only_list:
    remove_string = '{}_RMV'.format(only_attribute)
    x_train_is_canceled.drop([remove_string], axis=1, inplace=True)
    x_test_is_canceled.drop([remove_string], axis=1, inplace=True)

# print(x_train_is_canceled)
# print(x_test_is_canceled)
# print(y_train_is)

# ------

clf = TheRandomForest(x_train_is_canceled, y_train_is_canceled, x_test_is_canceled)
clf.v_fold_validate()

clf.train()
is_canceled_df = clf.predict()


hotel_adr = Dataset()

# From Hsu
# data_not_encoded = pd.concat([hotel_adr.get_feature(encode_target),hotel_adr.get_train_adr()],axis = 1)
# data_encoded = target_encode(data_not_encoded[encode_target],data_not_encoded['adr'])
# for col in data_encoded_col_name:
# 	data_encoded[col].fillna(data_encoded[col].mean(),inplace = True)
# hotel_adr.add_feature(data_encoded)
# hotel_adr.remove_feature(encode_target)

# ---remove only_feature---
# country_column = hotel_adr.get_feature(['country'])
remove_only_list = ['country', 'agent', 'company']
for only_attribute in remove_only_list:
    attribute_train_column = hotel_adr.get_train_column(only_attribute)
    attribute_test_column = hotel_adr.get_test_column(only_attribute)
    new_attribute_column = remove_only(hotel_adr.get_feature([only_attribute]), attribute_train_column, attribute_test_column)
    # print(new_country_column)
    # print(new_attribute_column[only_attribute].value_counts().loc['RMV'])
    hotel_adr.remove_feature([only_attribute])
    hotel_adr.add_feature(new_attribute_column)
# exit()
# -------------------------


x_train_adr = hotel_adr.get_train_dataset()
x_test_adr = hotel_adr.get_test_dataset()
y_train_adr = hotel_adr.get_train_adr()

for only_attribute in remove_only_list:
    remove_string = '{}_RMV'.format(only_attribute)
    x_train_adr.drop([remove_string], axis=1, inplace=True)
    x_test_adr.drop([remove_string], axis=1, inplace=True)


reg = TheRandomForestRegressor(x_train_adr, y_train_adr, x_test_adr)
reg.v_fold_validate()

reg.train()
adr_df = reg.predict()


predict_df = pd.concat([hotel_is_cancel.get_test_arrival_date(), is_canceled_df, adr_df, hotel_is_cancel.get_test_number_of_days()], axis=1)
predict(predict_df)


