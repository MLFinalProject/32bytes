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
	data_frame[column_name].values[data_frame[column_name] > threshold] = attribute_mean
	return data_frame
