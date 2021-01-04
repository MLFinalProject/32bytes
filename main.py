import pandas as pd

from dataset import Dataset
from classification import *
from regression import *
from feature_engineering import *
from predict import predict


hotel_is_cancel = Dataset()

# removing outliers
attribute_threshold_dict = {"adults":3, "babies":2, "children":3, "required_car_parking_spaces":2, "stays_in_week_nights":20, "stays_in_weekend_nights":10}
for key in attribute_threshold_dict:
    new_attribute_df = transfer_not_enough_data_to_mean(hotel_is_cancel.get_feature(key), attribute_threshold_dict[key])
    hotel_is_cancel.remove_feature(key)
    hotel_is_cancel.add_feature(new_attribute_df)
    print(hotel_is_cancel.get_feature(key))
exit()

room_feature = gen_room_feature(hotel_is_cancel.get_feature(['reserved_room_type', 'assigned_room_type']))
net_canceled_feature = gen_net_canceled_feature(hotel_is_cancel.get_feature(['previous_cancellations', 'previous_bookings_not_canceled']))
hotel_is_cancel.add_feature(room_feature)
hotel_is_cancel.add_feature(net_canceled_feature)
x_train_is_canceled = hotel_is_cancel.get_train_dataset()
x_test_is_canceled = hotel_is_cancel.get_test_dataset()
y_train_is_canceled = hotel_is_cancel.get_train_is_canceled()
dt = TheDecisionTree(x_train_is_canceled, y_train_is_canceled, x_test_is_canceled)
# dt.v_fold_validate()
dt.train()
is_canceled_df = dt.predict()


hotel_adr = Dataset()
x_train_adr = hotel_adr.get_train_dataset()
x_test_adr = hotel_adr.get_test_dataset()
y_train_adr = hotel_adr.get_train_adr()
dtregressor = TheDecisionTreeRegressor(x_train_adr, y_train_adr, x_test_adr)
# dtregressor.v_fold_validate()
dtregressor.train()
adr_df = dtregressor.predict()


predict_df = pd.concat([hotel_is_cancel.get_test_arrival_date(), is_canceled_df, adr_df, hotel_is_cancel.get_test_number_of_days()], axis=1)
predict(predict_df)


