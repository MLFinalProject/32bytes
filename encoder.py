import pandas as pd 
import numpy as np
import category_encoders #pip install category_encoders

def backward_difference_encode(input_df_x,input_df_y):
	input_df_x.columns = input_df_x.columns + '_encoded'
	result = pd.DataFrame()
	for x in input_df_x.columns:
		enc = category_encoders.BackwardDifferenceEncoder(cols = x)
		enc = enc.fit_transform(input_df_x[x],input_df_y)
		result[x] = enc.values.tolist()
	return result

def count_encode(input_df_x,input_df_y):
	input_df_x.columns = input_df_x.columns + '_encoded'
	enc = category_encoders.CountEncoder(cols = input_df_x.columns)
	return enc.fit_transform(input_df_x,input_df_y)

def helmert_encode(input_df_x,input_df_y):
	input_df_x.columns = input_df_x.columns + '_encoded'
	result = pd.DataFrame()
	for x in input_df_x.columns:
		enc = category_encoders.HelmertEncoder(cols = x)
		enc = enc.fit_transform(input_df_x[x],input_df_y)
		result[x] = enc.values.tolist()
	return result

def leave_one_out_encode(input_df_x,input_df_y):
	input_df_x.columns = input_df_x.columns + '_encoded'
	enc = category_encoders.LeaveOneOutEncoder(cols = input_df_x.columns)
	return enc.fit_transform(input_df_x,input_df_y)

def m_estimate_encode(input_df_x,input_df_y,y_threshold):
	input_df_x.columns = input_df_x.columns + '_encoded'
	y_label = input_df_y.columns[0]
	input_df_y[y_label] = (input_df_y[y_label] - y_threshold) > 0
	enc = category_encoders.MEstimateEncoder(cols = input_df_x.columns)
	return enc.fit_transform(input_df_x,input_df_y)

def one_hot_encode(input_df_x,input_df_y):
	input_df_x.columns = input_df_x.columns + '_encoded'
	result = pd.DataFrame()
	for x in input_df_x.columns:
		enc = category_encoders.OneHotEncoder(cols = x)
		enc = enc.fit_transform(input_df_x[x],input_df_y)
		result[x] = enc.values.tolist()
	return result
	
def polynomial_encode(input_df_x,input_df_y):
	input_df_x.columns = input_df_x.columns + '_encoded'
	enc = category_encoders.PolynomialEncoder(cols = input_df_x.columns)
	return enc.fit_transform(input_df_x,input_df_y)

def target_encode(input_df_x,input_df_y):
	input_df_x.columns = input_df_x.columns + '_encoded'
	enc = category_encoders.TargetEncoder(cols = input_df_x.columns)
	return enc.fit_transform(input_df_x,input_df_y)
	
def weight_of_evidence_encode(input_df_x,input_df_y,y_threshold):
	input_df_x.columns = input_df_x.columns + '_encoded'
	y_label = input_df_y.columns[0]
	input_df_y[y_label] = (input_df_y[y_label] - y_threshold) > 0
	enc = category_encoders.WOEEncoder(cols = input_df_x.columns)
	return enc.fit_transform(input_df_x,input_df_y)