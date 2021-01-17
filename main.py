import pandas as pd

from dataset import Dataset
from classification import *
from regression import *
from feature_engineering import *
from predict import predict

hotel_is_cancel = Dataset()

room_feature = gen_room_feature(hotel_is_cancel.get_feature(['reserved_room_type', 'assigned_room_type']))
net_canceled_feature = gen_net_canceled_feature(hotel_is_cancel.get_feature(['previous_cancellations', 'previous_bookings_not_canceled']))
hotel_is_cancel.add_feature(room_feature)
hotel_is_cancel.add_feature(net_canceled_feature)

# ---remove only_feature---
# country_column = hotel_is_cancel.get_feature(['country'])
remove_only_list = ['country', 'agent']
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
hotel_is_cancel.remove_feature(['company'])


# ---Remove only---

hotel_is_cancel.remove_feature(['arrival_date_year'])

x_train_is_canceled = hotel_is_cancel.get_train_dataset()
x_test_is_canceled = hotel_is_cancel.get_test_dataset()
y_train_is_canceled = hotel_is_cancel.get_train_is_canceled()


for only_attribute in remove_only_list:
    remove_string = '{}_RMV'.format(only_attribute)
    x_train_is_canceled.drop([remove_string], axis=1, inplace=True)
    x_test_is_canceled.drop([remove_string], axis=1, inplace=True)


# ------

# deposit_type_column = x_test_is_canceled[['deposit_type_Non Refund']]
# print(deposit_type_column)
# exit()


clf = TheRandomForest(x_train_is_canceled, y_train_is_canceled, x_test_is_canceled)
# clf.v_fold_validate()

clf.train()
is_canceled_df = clf.predict()
# print(hotel_is_cancel.get_feature(['deposit_type']))
# is_canceled_df
# print(is_canceled_df)
# deposit_type_column = hotel_is_cancel.train_test_df.eq(hotel_is_cancel.data_label["test"])[['deposit_type']]
deposit_type_column = x_test_is_canceled[['deposit_type_Non Refund']]
# print(deposit_type_column)
# print(is_canceled_df)
print(is_canceled_df['is_canceled'].value_counts().loc[1])
is_canceled_df['is_canceled'].values[deposit_type_column['deposit_type_Non Refund'] == 1] = 1
print(is_canceled_df['is_canceled'].value_counts().loc[1])
# data_frame['country'].values[data_frame['country'].isin(only_country)] = 'RMV'
print(is_canceled_df)
# exit()


hotel_adr = Dataset()

# ---remove only_feature---
# country_column = hotel_adr.get_feature(['country'])
remove_only_list = ['country', 'agent']
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
hotel_adr.remove_feature(['company'])
hotel_adr.remove_feature(['arrival_date_year'])

x_train_adr = hotel_adr.get_train_dataset()
x_test_adr = hotel_adr.get_test_dataset()
y_train_adr = hotel_adr.get_train_adr()

for only_attribute in remove_only_list:
    remove_string = '{}_RMV'.format(only_attribute)
    x_train_adr.drop([remove_string], axis=1, inplace=True)
    x_test_adr.drop([remove_string], axis=1, inplace=True)


reg = TheRandomForestRegressor(x_train_adr, y_train_adr, x_test_adr)
# reg.v_fold_validate()

reg.train()
adr_df = reg.predict()


predict_df = pd.concat([hotel_is_cancel.get_test_arrival_date(), is_canceled_df, adr_df, hotel_is_cancel.get_test_number_of_days()], axis=1)
predict(predict_df)


