import numpy as np
from encoder import *
from feature_engineering import *
from dataset import Dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from predict import predict

target_encode_item = ['deposit_type','customer_type']
count_encode_item = ['agent','country','market_segment']

hotel_is_cancel = Dataset()
room_feature = gen_room_feature(hotel_is_cancel.get_feature(['reserved_room_type', 'assigned_room_type']))
net_canceled_feature = gen_net_canceled_feature(hotel_is_cancel.get_feature(['previous_cancellations', 'previous_bookings_not_canceled']))
hotel_is_cancel.add_feature(room_feature)
hotel_is_cancel.add_feature(net_canceled_feature)

data_not_encoded = pd.concat([hotel_is_cancel.get_feature(target_encode_item),hotel_is_cancel.get_train_is_canceled()],axis = 1)
data_encoded = target_encode(data_not_encoded[target_encode_item],data_not_encoded['is_canceled'])
for col in data_encoded.columns:
	data_encoded[col].fillna(data_encoded[col].mean(),inplace = True)
hotel_is_cancel.add_feature(data_encoded)
#hotel_is_cancel.remove_feature(target_encode_item)

data_not_encoded = pd.concat([hotel_is_cancel.get_feature(count_encode_item),hotel_is_cancel.get_train_is_canceled()],axis = 1)
data_encoded = target_encode(data_not_encoded[count_encode_item],data_not_encoded['is_canceled'])
for col in data_encoded.columns:
	data_encoded[col].fillna(data_encoded[col].mean(),inplace = True)
hotel_is_cancel.add_feature(data_encoded)
#hotel_is_cancel.remove_feature(count_encode_item)

x_train_is_canceled = hotel_is_cancel.get_train_dataset()
x_test_is_canceled = hotel_is_cancel.get_test_dataset()
y_train_is_canceled = hotel_is_cancel.get_train_is_canceled()

x_train, x_val, y_train, y_val = train_test_split(x_train_is_canceled,y_train_is_canceled,random_state = 1126)

model = SelectFromModel(RandomForestClassifier(min_impurity_decrease=1e-6,max_depth=10 ,n_estimators=128, random_state = 6174, n_jobs = -1))
model = model.fit(x_train,y_train)

x_train_is_canceled = model.transform(x_train_is_canceled)
x_test_is_canceled = model.transform(x_test_is_canceled)
x_train = model.transform(x_train)
x_val = model.transform(x_val)

model = RandomForestClassifier(min_impurity_decrease=1e-6,max_depth=10 ,n_estimators=128, random_state = 6174, n_jobs = -1)
model = model.fit(x_train,y_train)
print(f'acc = {model.score(x_val,y_val)}')

model = model.fit(x_train_is_canceled,y_train_is_canceled)
is_canceled_df = pd.DataFrame(model.predict(x_test_is_canceled), columns = ['is_canceled'])

print(is_canceled_df)

hotel_adr = Dataset()
room_feature = gen_room_feature(hotel_adr.get_feature(['reserved_room_type', 'assigned_room_type']))
net_canceled_feature = gen_net_canceled_feature(hotel_adr.get_feature(['previous_cancellations', 'previous_bookings_not_canceled']))
hotel_adr.add_feature(room_feature)
hotel_adr.add_feature(net_canceled_feature)

data_not_encoded = pd.concat([hotel_adr.get_feature(target_encode_item),hotel_adr.get_train_adr()],axis = 1)
data_encoded = target_encode(data_not_encoded[target_encode_item],data_not_encoded['adr'])
for col in data_encoded.columns:
	data_encoded[col].fillna(data_encoded[col].mean(),inplace = True)
hotel_adr.add_feature(data_encoded)
hotel_is_cancel.remove_feature(target_encode_item)

data_not_encoded = pd.concat([hotel_adr.get_feature(count_encode_item),hotel_adr.get_train_adr()],axis = 1)
data_encoded = target_encode(data_not_encoded[count_encode_item],data_not_encoded['adr'])
for col in data_encoded.columns:
	data_encoded[col].fillna(data_encoded[col].mean(),inplace = True)
hotel_adr.add_feature(data_encoded)
hotel_is_cancel.remove_feature(count_encode_item)

x_train_adr = hotel_adr.get_train_dataset()
x_test_adr = hotel_adr.get_test_dataset()
y_train_adr = hotel_adr.get_train_adr()

x_train, x_val, y_train, y_val = train_test_split(x_train_adr,y_train_adr,random_state = 1126)

model = SelectFromModel(RandomForestRegressor(min_impurity_decrease=0.001, max_features=0.4, min_samples_leaf = 2, n_estimators=128, random_state = 6174, bootstrap=True, n_jobs = -1))
model = model.fit(x_train,y_train)

x_train_adr = model.transform(x_train_adr)
x_test_adr = model.transform(x_test_adr)
x_train = model.transform(x_train)
x_val = model.transform(x_val)

model = RandomForestRegressor(min_impurity_decrease=0.001, max_features=0.55, min_samples_leaf = 2, n_estimators=128, random_state = 6174, bootstrap=True, n_jobs = -1)
model = model.fit(x_train,y_train)
print(f'acc = {model.score(x_val,y_val)}')

model = model.fit(x_train_adr,y_train_adr)
adr_df = pd.DataFrame(model.predict(x_test_adr), columns = ['adr'])

print(adr_df)

predict_df = pd.concat([hotel_is_cancel.get_test_arrival_date(), is_canceled_df, adr_df, hotel_is_cancel.get_test_number_of_days()], axis=1)
predict(predict_df)