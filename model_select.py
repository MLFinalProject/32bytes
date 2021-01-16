import numpy as np
from encoder import *
from feature_engineering import *
from dataset import Dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVR
from predict import predict
import random 

encode_target = ['hotel','agent','arrival_date_day_of_month','assigned_room_type','customer_type','deposit_type','distribution_channel','market_segment','meal','reserved_room_type']
target_encode_item = ['deposit_type','hotel','arrival_date_month','reserved_room_type','assigned_room_type','arrival_date_day_of_month','meal']
count_encode_item = ['agent','country','market_segment','customer_type','distribution_channel']
remove_item = ['arrival_date_year','babies','company','booking_changes']

#target_encode_item = encode_target
count_encode_item = encode_target

hotel_is_cancel = Dataset()
room_feature = gen_room_feature(hotel_is_cancel.get_feature(['reserved_room_type', 'assigned_room_type']))
net_canceled_feature = gen_net_canceled_feature(hotel_is_cancel.get_feature(['previous_cancellations', 'previous_bookings_not_canceled']))
hotel_is_cancel.add_feature(room_feature)
hotel_is_cancel.add_feature(net_canceled_feature)
hotel_is_cancel.remove_feature(remove_item)

# data_not_encoded = pd.concat([hotel_is_cancel.get_feature(target_encode_item),hotel_is_cancel.get_train_is_canceled()],axis = 1)
# data_encoded = target_encode(data_not_encoded[target_encode_item],data_not_encoded['is_canceled'])
# for col in data_encoded.columns:
	# data_encoded[col].fillna(data_encoded[col].mean(),inplace = True)
# hotel_is_cancel.add_feature(data_encoded)
# hotel_is_cancel.remove_feature(target_encode_item)

# data_not_encoded = pd.concat([hotel_is_cancel.get_feature(count_encode_item),hotel_is_cancel.get_train_is_canceled()],axis = 1)
# data_encoded = count_encode(data_not_encoded[count_encode_item],data_not_encoded['is_canceled'])
# for col in data_encoded.columns:
	# data_encoded[col].fillna(data_encoded[col].mean(),inplace = True)
# hotel_is_cancel.add_feature(data_encoded)
# hotel_is_cancel.remove_feature(count_encode_item)

x_train_is_canceled = hotel_is_cancel.get_train_dataset()
x_test_is_canceled = hotel_is_cancel.get_test_dataset()
y_train_is_canceled = hotel_is_cancel.get_train_is_canceled()
print(x_train_is_canceled.shape)

x_train, x_val, y_train, y_val = train_test_split(x_train_is_canceled,y_train_is_canceled,random_state = random.seed())

model = RandomForestClassifier(min_impurity_decrease=1e-6,max_depth=20 ,n_estimators=128, random_state = 5498, n_jobs = -1,oob_score = True)
model = SelectFromModel(model)
model = model.fit(x_train,y_train)

x_train_is_canceled = model.transform(x_train_is_canceled)
x_test_is_canceled = model.transform(x_test_is_canceled)
x_train = model.transform(x_train)
x_val = model.transform(x_val)

model = RandomForestClassifier(min_impurity_decrease=1e-6,max_depth=20 ,n_estimators=128, random_state = 5497, n_jobs = -1,oob_score = True)
model = model.fit(x_train,y_train)
print(f'val acc = {model.score(x_val,y_val)}')
model = model.fit(x_train_is_canceled,y_train_is_canceled)
print(f'train acc = {model.score(x_train_is_canceled,y_train_is_canceled)}')

model = model.fit(x_train_is_canceled,y_train_is_canceled)
is_canceled_df = pd.DataFrame(model.predict(x_test_is_canceled), columns = ['is_canceled'])

print(is_canceled_df)

target_encode_item = ['hotel','arrival_date_month','meal','deposit_type','assigned_room_type','customer_type','reserved_room_type','distribution_channel']
count_encode_item = ['agent','market_segment']
remove_item = ['arrival_date_year','company','country']

hotel_adr = Dataset()
room_feature = gen_room_feature(hotel_adr.get_feature(['reserved_room_type', 'assigned_room_type']))
net_canceled_feature = gen_net_canceled_feature(hotel_adr.get_feature(['previous_cancellations', 'previous_bookings_not_canceled']))
hotel_adr.add_feature(room_feature)
hotel_adr.add_feature(net_canceled_feature)
hotel_adr.remove_feature(remove_item)

# data_not_encoded = pd.concat([hotel_adr.get_feature(target_encode_item),hotel_adr.get_train_adr()],axis = 1)
# data_encoded = target_encode(data_not_encoded[target_encode_item],data_not_encoded['adr'])
# for col in data_encoded.columns:
	# data_encoded[col].fillna(data_encoded[col].mean(),inplace = True)
# hotel_adr.add_feature(data_encoded)
# hotel_adr.remove_feature(target_encode_item)

# data_not_encoded = pd.concat([hotel_adr.get_feature(count_encode_item),hotel_adr.get_train_adr()],axis = 1)
# data_encoded = count_encode(data_not_encoded[count_encode_item],data_not_encoded['adr'])
# for col in data_encoded.columns:
	# data_encoded[col].fillna(data_encoded[col].mean(),inplace = True)
# hotel_adr.add_feature(data_encoded)
# hotel_adr.remove_feature(count_encode_item)

x_train_adr = hotel_adr.get_train_dataset()
x_test_adr = hotel_adr.get_test_dataset()
y_train_adr = hotel_adr.get_train_adr()

x_train, x_val, y_train, y_val = train_test_split(x_train_adr,y_train_adr,random_state = random.seed())

model = SelectFromModel(RandomForestRegressor(min_impurity_decrease=0.001, max_features=0.55, min_samples_leaf = 2, n_estimators=128, random_state = 6174, bootstrap=True, n_jobs = -1,oob_score = True))
model = model.fit(x_train,y_train)

x_train_adr = model.transform(x_train_adr)
x_test_adr = model.transform(x_test_adr)
x_train = model.transform(x_train)
x_val = model.transform(x_val)
print(x_train_adr.shape)

model = RandomForestRegressor(min_impurity_decrease=0.001, max_features= 0.45, min_samples_leaf = 1, n_estimators=128, random_state = 6174, bootstrap=True, n_jobs = -1,oob_score = True)
model = model.fit(x_train,y_train)
print(f'val acc = {model.score(x_val,y_val)}')

model = model.fit(x_train_adr,y_train_adr)
print(f'train acc = {model.score(x_train_adr,y_train_adr)}')
adr_df = pd.DataFrame(model.predict(x_test_adr), columns = ['adr'])

print(adr_df)

predict_df = pd.concat([hotel_is_cancel.get_test_arrival_date(), is_canceled_df, adr_df, hotel_is_cancel.get_test_number_of_days()], axis=1)
predict(predict_df)