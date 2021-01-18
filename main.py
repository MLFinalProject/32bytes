import pandas as pd

from dataset import Dataset
from classification import *
from regression import *
from feature_engineering import *
from predict import predict
from encoder import *


# encode_target = ['agent']

hotel_is_cancel = Dataset()

# room_feature = gen_room_feature(hotel_is_cancel.get_feature(['reserved_room_type', 'assigned_room_type']))
# net_canceled_feature = gen_net_canceled_feature(hotel_is_cancel.get_feature(['previous_cancellations', 'previous_bookings_not_canceled']))
# hotel_is_cancel.add_feature(room_feature)
# hotel_is_cancel.add_feature(net_canceled_feature)

# ---remove only_feature---
# country_column = hotel_is_cancel.get_feature(['country'])


# Remove only
remove_only_list = []
for only_attribute in remove_only_list:
    attribute_train_column = hotel_is_cancel.get_train_column(only_attribute)
    attribute_test_column = hotel_is_cancel.get_test_column(only_attribute)
    new_attribute_column = remove_only(hotel_is_cancel.get_feature([only_attribute]), attribute_train_column, attribute_test_column)
    hotel_is_cancel.remove_feature([only_attribute])
    hotel_is_cancel.add_feature(new_attribute_column)

# data_not_encoded = pd.concat([hotel_is_cancel.get_feature(encode_target),hotel_is_cancel.get_train_is_canceled()],axis = 1)
# data_encoded = target_encode(data_not_encoded[encode_target],data_not_encoded['is_canceled'])
# data_encoded_col_name = data_encoded.columns
# for col in data_encoded_col_name:
#   data_encoded[col].fillna(data_encoded[col].mean(),inplace = True)
# hotel_is_cancel.add_feature(data_encoded)

# exit()
# -------------------------
# remove feature
# hotel_is_cancel.remove_feature(['company'])
# hotel_is_cancel.remove_feature(['arrival_date_year'])

# x_train_is_canceled = hotel_is_cancel.get_train_dataset()
# x_test_is_canceled = hotel_is_cancel.get_test_dataset()
# y_train_is_canceled = hotel_is_cancel.get_train_is_canceled()

# remove_only_list = ['country']

# for only_attribute in remove_only_list:
#     remove_string = '{}_RMV'.format(only_attribute)
#     x_train_is_canceled.drop([remove_string], axis=1, inplace=True)
#     x_test_is_canceled.drop([remove_string], axis=1, inplace=True)


# ------

hotel_adr = Dataset()

# ---remove only_feature---
# country_column = hotel_adr.get_feature(['country'])
# remove_only_list = ['country', 'agent']
for only_attribute in remove_only_list:
    attribute_train_column = hotel_adr.get_train_column(only_attribute)
    attribute_test_column = hotel_adr.get_test_column(only_attribute)
    new_attribute_column = remove_only(hotel_adr.get_feature([only_attribute]), attribute_train_column, attribute_test_column)
    hotel_adr.remove_feature([only_attribute])
    hotel_adr.add_feature(new_attribute_column)
# exit()
# -------------------------

# data_not_encoded = pd.concat([hotel_adr.get_feature(encode_target),hotel_adr.get_train_adr()],axis = 1)
# data_encoded = target_encode(data_not_encoded[encode_target],data_not_encoded['adr'])
# data_encoded_col_name = data_encoded.columns
# for col in data_encoded_col_name:
#   data_encoded[col].fillna(data_encoded[col].mean(),inplace = True)
# hotel_adr.add_feature(data_encoded)
attribute_list = list(hotel_is_cancel.train_test_df.columns.values)
attribute_list.remove('dataset')
attribute_list.remove('arrival_date_month')


for attribute in attribute_list[2:]:
    write_string_list = []
    write_string_list.append(attribute)
    # remove_feature = "None"
    print("will drop {}".format(attribute))
    attribute_is_cancel = hotel_is_cancel.get_feature([attribute])
    attribute_adr = hotel_adr.get_feature([attribute])
    hotel_is_cancel.remove_feature([attribute])
    hotel_adr.remove_feature([attribute])
    # hotel_adr.remove_feature(['arrival_date_year'])

    x_train_is_canceled = hotel_is_cancel.get_train_dataset()
    x_test_is_canceled = hotel_is_cancel.get_test_dataset()
    y_train_is_canceled = hotel_is_cancel.get_train_is_canceled()

    x_train_adr = hotel_adr.get_train_dataset()
    x_test_adr = hotel_adr.get_test_dataset()
    y_train_adr = hotel_adr.get_train_adr()


    # remove_only_list = ['country']
    for only_attribute in remove_only_list:
        remove_string = '{}_RMV'.format(only_attribute)
        x_train_adr.drop([remove_string], axis=1, inplace=True)
        x_test_adr.drop([remove_string], axis=1, inplace=True)


    # with open('validation.csv', 'a', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow([attribute])

    # model selection:
    seed_num = 112

    clf = TheRandomForest(x_train_is_canceled, y_train_is_canceled, x_test_is_canceled, seed = seed_num) #seed = 112, 1126, 6174
    # clf = TheGradientBoost(x_train_is_canceled, y_train_is_canceled, x_test_is_canceled, seed = 112) #seed = 112, 1126, 6174
    # clf = TheDecisionTree(x_train_is_canceled, y_train_is_canceled, x_test_is_canceled, seed = 112) #seed = 112, 1126, 6174
    # clf = TheLogisticRegression(x_train_is_canceled, y_train_is_canceled, x_test_is_canceled, seed = 112) #seed = 112, 1126, 6174

    write_string_list += clf.monthly_validate(seed = seed_num)
    clf.train()
    is_canceled_df = clf.predict()


    reg = TheRandomForestRegressor(x_train_adr, y_train_adr, x_test_adr, seed = seed_num) #seed = 112, 1126, 6174
    # reg = TheGradientBoostingRegressor(x_train_adr, y_train_adr, x_test_adr, seed = 112) #seed = 112, 1126, 6174
    # reg = TheDecisionTreeRegressor(x_train_adr, y_train_adr, x_test_adr, seed = 112) #seed = 112, 1126, 6174
    # reg = TheLinearRegression(x_train_adr, y_train_adr, x_test_adr) #no seed required

    write_string_list += reg.monthly_validate(seed = seed_num)
    reg.train()
    adr_df = reg.predict()


    predict_df = pd.concat([hotel_is_cancel.get_test_arrival_date(), is_canceled_df, adr_df, hotel_is_cancel.get_test_number_of_days()], axis=1)
    file_name = 'out_put/out_put_without_{}_{}.csv'.format(attribute, seed_num)
    predict(predict_df, file_name)

    with open('validation.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(write_string_list)

    hotel_is_cancel.add_feature(attribute_is_cancel)
    hotel_adr.add_feature(attribute_adr)


