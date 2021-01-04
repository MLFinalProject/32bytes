from dataset import Dataset
from classification import *
from regression import *
from feature_engineering import *


hotel_is_cancel = Dataset()
room_feature = gen_room_feature(hotel_is_cancel.get_feature(['reserved_room_type', 'assigned_room_type']))
net_canceled_feature = gen_net_canceled_feature(hotel_is_cancel.get_feature(['previous_cancellations', 'previous_bookings_not_canceled']))
hotel_is_cancel.add_feature(room_feature)
hotel_is_cancel.add_feature(net_canceled_feature)
x_train = hotel_is_cancel.get_train_dataset()
x_test = hotel_is_cancel.get_test_dataset()
y_train = hotel_is_cancel.get_train_is_canceled()
decisiontree = TheDecisionTree(x_train, y_train, x_test)
decisiontree.train()
decisiontree.v_fold_validate()
# from feature_engineering import *
# from dataset import Dataset
# hotel = Dataset()
# room_feature = gen_room_feature(hotel.get_feature(['reserved_room_type', 'assigned_room_type']))
# net_canceled_feature = gen_net_canceled_feature(hotel.get_feature(['previous_cancellations', 'previous_bookings_not_canceled']))
# hotel.add_feature(room_feature)
# hotel.add_feature(net_canceled_feature)
# hotel.remove_feature(['children', 'stays_in_weekend_nights'])
# x_train = hotel.get_train_dataset()
# x_test = hotel.get_test_dataset()
# y_train = hotel.get_train_is_canceled()
# decisiontree = TheDecisionTree(x_train, y_train, x_test)
# decisiontree.train()
# print(decisiontree.v_fold_validate())
