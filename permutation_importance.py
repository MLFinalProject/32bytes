import matplotlib.pyplot as plt
import numpy as np
from encoder import *
from feature_engineering import *
from dataset import Dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

hotel_is_cancel = Dataset()
encode_target = ['hotel','agent','arrival_date_day_of_month','arrival_date_year','assigned_room_type','company','country','customer_type','deposit_type','distribution_channel','market_segment','meal','reserved_room_type']

room_feature = gen_room_feature(hotel_is_cancel.get_feature(['reserved_room_type', 'assigned_room_type']))
net_canceled_feature = gen_net_canceled_feature(hotel_is_cancel.get_feature(['previous_cancellations', 'previous_bookings_not_canceled']))
hotel_is_cancel.add_feature(room_feature)
hotel_is_cancel.add_feature(net_canceled_feature)

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

x_train, x_val, y_train, y_val = train_test_split(x_train_is_canceled,y_train_is_canceled,random_state = 0)

model = RandomForestClassifier(min_impurity_decrease=1e-6,max_depth=10 ,n_estimators=128, random_state = 6174, n_jobs = -1)
model = model.fit(x_train,y_train)
print(f'acc = {model.score(x_val,y_val)}')

r = permutation_importance(model,x_val,y_val,n_repeats = 30,random_state = 0)

arr = np.append(r.importances_mean.reshape(1,41),r.importances_std.reshape(1,41),axis = 0)
#print(arr.shape)
feature_df = pd.DataFrame(arr,columns = x_val.columns)
feature_df.index = ['mean','std']
feature_df = feature_df.sort_values(by= 'mean',axis = 1, ascending = False)
feature_df.to_csv('./feature.csv')

# for i in range(len(r.importances_mean)):
	# print(f'mean: {r.importances_mean[i]:.4E}, std: {r.importances_std[i]:.4E} ({x_val.columns[i]})')
