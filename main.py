import pandas as pd

from dataset import Dataset
from classification import *
from regression import *
from feature_engineering import *
from predict import predict
from encoder import *


encode_target = ['hotel','agent','arrival_date_day_of_month','arrival_date_year','assigned_room_type','company','country','customer_type','deposit_type','distribution_channel','market_segment','meal','reserved_room_type']
#encode_target = ['country']

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
#hotel_is_cancel.remove_feature(['country','arrival_date_year'])

data_not_encoded = pd.concat([hotel_is_cancel.get_feature(encode_target),hotel_is_cancel.get_train_is_canceled()],axis = 1)
data_encoded = count_encode(data_not_encoded[encode_target],data_not_encoded['is_canceled'])
data_encoded_col_name = data_encoded.columns
for col in data_encoded_col_name:
	data_encoded[col].fillna(data_encoded[col].mean(),inplace = True)
hotel_is_cancel.add_feature(data_encoded)
hotel_is_cancel.remove_feature(encode_target)

x_train_is_canceled = hotel_is_cancel.get_train_dataset()
x_test_is_canceled = hotel_is_cancel.get_test_dataset()
y_train_is_canceled = hotel_is_cancel.get_train_is_canceled()
clf = TheRandomForest(x_train_is_canceled, y_train_is_canceled, x_test_is_canceled)
clf.v_fold_validate()

clf.train()
is_canceled_df = clf.predict()


hotel_adr = Dataset()
#hotel_adr.remove_feature(['country','arrival_date_year'])

data_not_encoded = pd.concat([hotel_adr.get_feature(encode_target),hotel_adr.get_train_adr()],axis = 1)
data_encoded = count_encode(data_not_encoded[encode_target],data_not_encoded['adr'])
for col in data_encoded_col_name:
	data_encoded[col].fillna(data_encoded[col].mean(),inplace = True)
hotel_adr.add_feature(data_encoded)
hotel_adr.remove_feature(encode_target)

x_train_adr = hotel_adr.get_train_dataset()
x_test_adr = hotel_adr.get_test_dataset()
y_train_adr = hotel_adr.get_train_adr()
reg = TheRandomForestRegressor(x_train_adr, y_train_adr, x_test_adr)
reg.v_fold_validate()

reg.train()
adr_df = reg.predict()


predict_df = pd.concat([hotel_is_cancel.get_test_arrival_date(), is_canceled_df, adr_df, hotel_is_cancel.get_test_number_of_days()], axis=1)
predict(predict_df)


