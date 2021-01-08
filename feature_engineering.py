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
