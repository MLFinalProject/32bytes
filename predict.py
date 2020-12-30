import pandas as pd
import numpy as np
from dataset import *

def predict(input_df,filename='output.csv'):
	#p = Dataset()
	#input_df = pd.concat([p.get_arrival_date(), p.get_is_canceled(), p.get_adr(), p.get_number_of_days()], axis=1)

	input_df['is_canceled'] = input_df['is_canceled'] * (-1) + 1
	input_df['total_adr'] = input_df['is_canceled'] * input_df['adr']* input_df['number_of_days']
	input_df = input_df[['arrival_date','total_adr']] 
	input_df = input_df.groupby('arrival_date').sum()
	input_df['label'] = (input_df['total_adr'] / 10000).astype(int) 
	input_df = input_df[['label']] 
	input_df.to_csv(filename)

