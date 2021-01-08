import pandas as pd
import numpy as np

def gen_room_feature(data_frame):
	data_frame = data_frame.assign(room=0)
	data_frame.loc[ data_frame['reserved_room_type'] == data_frame['assigned_room_type'] , 'room'] = 1
	return data_frame[['room']]

def gen_net_canceled_feature(data_frame):
	data_frame = data_frame.assign(net_canceled=0)
	data_frame.loc[ data_frame['previous_cancellations'] > data_frame['previous_bookings_not_canceled'] , 'net_canceled'] = 1
	return data_frame[['net_canceled']]

def transfer_not_enough_data_to_mean(data_frame, threshold):
	attribute_mean = round(data_frame.mean())
	column_name = data_frame.columns[0]
	data_frame[column_name].values[data_frame[column_name] > threshold] = attribute_mean + 100
	return data_frame

def absolute_peak_transform(data_frame, peak):
	column_name = data_frame.columns[0]
	df = abs(data_frame.loc[:, column_name] - peak)
	# print(df)
	return df
	# exit()
	# # data_frame.loc[:, [column_name]] += 1
	# print(data_frame[column_name])
	# exit()
	# return data_frame

def adults_only(data_frame):
	data_frame = data_frame.assign(adults_only=0)
	column_name = data_frame.columns
	data_frame.loc[ (data_frame['adults'] > 0) & (data_frame['children'] == 0) & (data_frame['babies'] == 0), 'adults_only'] = 1

	# print(column_name[1])
	print(data_frame)
	print(data_frame['adults_only'])

def country_remove_only(data_frame, data_frame_train, data_frame_test):
	train_country = data_frame_train.to_numpy().squeeze()
	test_country = data_frame_test.to_numpy().squeeze()
	train_only_country = list(set(train_country) - set(test_country))
	test_only_country = list(set(test_country) - set(train_country))
	only_country = train_only_country + test_only_country
	print(train_only_country)
	print(test_only_country)
	print(only_country)
	# print(test_only_country)
	# print(data_frame)
	# # exit()
	# print(type(data_frame['country']))
	data_frame['country'].values[data_frame['country'].isin(only_country)] = 'RMV'
	# data_frame['country'].values[data_frame['country'].isnull()] = 'RMV'
	# data_frame.loc[data_frame['country'].isin(train_only_country), 'country'] = 'RMV'
	# print(data_frame['country'].value_counts())
	# Totally there are 103 'RMV' in train
	# print(data_frame['country'].value_counts().loc['RMV'])
	# print(type(data_frame['country'].value_counts()))
	return data_frame


def remove_only(data_frame, data_frame_train, data_frame_test):
	attribute = list(data_frame.columns.values)[0]
	train_attribute = data_frame_train.to_numpy().squeeze()
	test_attribute = data_frame_test.to_numpy().squeeze()
	train_only_attribute = list(set(train_attribute) - set(test_attribute))
	test_only_attribute = list(set(test_attribute) - set(train_attribute))
	only_attribute = train_only_attribute + test_only_attribute
	# print(train_only_attribute)
	# print(test_only_attribute)
	# print(only_attribute)
	# print(test_only_attribute)
	# print(data_frame)
	# # exit()
	# print(type(data_frame['attribute']))
	data_frame[attribute].values[data_frame[attribute].isin(only_attribute)] = 'RMV'

	return data_frame
	
