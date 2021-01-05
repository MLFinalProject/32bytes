import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import time

class Regression:
	def __init__(self, x_train, y_train, x_test):
		self.x_train = x_train
		self.y_train = np.ravel(y_train)
		self.x_test = x_test
		
	def v_fold_validate(self):
		self.start_time = time.time()
		self.x_val_train, self.x_val_test, self.y_val_train, self.y_val_test = train_test_split(self.x_train, self.y_train, test_size = 0.2, random_state = 390625)
    
	def train(self):
		print('\n---adr Training---')
		self.start_time = time.time()

	def predict(self):
		print('\n---adr Predicting---')
		self.start_time = time.time()
		
class TheLinearRegression(Regression):
	def __init__(self, x_train, y_val_train, x_test):
		super().__init__(x_train, y_train, x_test)
		self.reg = LinearRegression()
	
	def v_fold_validate(self):
		super().v_fold_validate()
		self.reg = self.reg.fit(self.x_val_train, self.y_val_train)
		train_acc = self.reg.score(self.x_val_train, self.y_val_train)
		test_acc = self.reg.score(self.x_val_test, self.y_val_test)
		print('---Cross-Validation Testing---')
		print(f'Training Accuracy of our model is: {train_acc}')
		print(f'Cross-Validation Test Accuracy of our model is: {test_acc}')
		print(f'adr validation done in {time.time()-self.start_time:.3f}(s).')
	
	def train(self):
		super().train()
		self.reg = self.reg.fit(self.x_train,self.y_train)
		train_acc = self.reg.score(self.x_train,self.y_train)
		print(f'Training Accuracy of our model is: {train_acc:.3f}')
		print(f'adr training done in {time.time()-self.start_time:.3f}(s).')
		
	def predict(self):
		super().predict()
		predicts = pd.DataFrame(self.reg.predict(self.x_test), columns = ['adr'])
		print(f'adr prediction done in {time.time()-self.start_time:.3f}(s).')
		return predicts

class TheDecisionTreeRegressor(Regression):
	def __init__(self, x_train, y_train, x_test):
		super().__init__(x_train, y_train, x_test)
		self.reg = DecisionTreeRegressor(random_state = 5)
	
	def v_fold_validate(self):
		super().v_fold_validate()
		self.reg = self.reg.fit(self.x_val_train, self.y_val_train)
		train_acc = self.reg.score(self.x_val_train, self.y_val_train)
		test_acc = self.reg.score(self.x_val_test, self.y_val_test)
		print('---Cross-Validation Testing---')
		print(f'Training Accuracy of our model is: {train_acc:.3f}')
		print(f'Cross-Validation Test Accuracy of our model is: {test_acc:.3f}')
		print(f'adr validation done in {time.time()-self.start_time:.3f}(s).')
	
	def train(self):
		super().train()
		self.reg = self.reg.fit(self.x_train,self.y_train)
		train_acc = self.reg.score(self.x_train,self.y_train)
		print(f'Training Accuracy of our model is: {train_acc:.3f}')
		print(f'adr training done in {time.time()-self.start_time:.3f}(s).')
		
	def predict(self):
		super().predict()
		predicts = pd.DataFrame(self.reg.predict(self.x_test), columns = ['adr'])
		print(f'adr prediction done in {time.time()-self.start_time:.3f}(s).')
		return predicts

class TheGradientBoostingRegressor(Regression):
	def __init__(self, x_train, y_train, x_test):
		super().__init__(x_train, y_train, x_test)
		self.reg = GradientBoostingRegressor(random_state = 0, n_estimators = 100, loss='lad', max_depth = 4)
	
	def v_fold_validate(self):
		super().v_fold_validate()
		self.reg = self.reg.fit(self.x_val_train, self.y_val_train)
		train_acc = self.reg.score(self.x_val_train, self.y_val_train)
		test_acc = self.reg.score(self.x_val_test, self.y_val_test)
		print('---Cross-Validation Testing---')
		print(f'Training Accuracy of our model is: {train_acc:.3f}')
		print(f'Cross-Validation Test Accuracy of our model is: {test_acc:.3f}')
		print(f'adr validation done in {time.time()-self.start_time:.3f}(s).')
	
	def train(self):
		super().train()
		self.reg = self.reg.fit(self.x_train,self.y_train)
		train_acc = self.reg.score(self.x_train,self.y_train)
		print(f'Training Accuracy of our model is: {train_acc:.3f}')
		print(f'adr training done in {time.time()-self.start_time:.3f}(s).')
		
	def predict(self):
		super().predict()
		predicts = pd.DataFrame(self.reg.predict(self.x_test), columns = ['adr'])
		print(f'adr prediction done in {time.time()-self.start_time:.3f}(s).')
		return predicts

class TheRandomForestRegressor(Regression):
	def __init__(self, x_train, y_train, x_test):
		super().__init__(x_train, y_train, x_test)
		self.reg = RandomForestRegressor(random_state = 0, n_estimators = 100, n_jobs = -1)
	
	def v_fold_validate(self):
		super().v_fold_validate()
		self.reg = self.reg.fit(self.x_val_train, self.y_val_train)
		train_acc = self.reg.score(self.x_val_train, self.y_val_train)
		test_acc = self.reg.score(self.x_val_test, self.y_val_test)
		print('---Cross-Validation Testing---')
		print(f'Training Accuracy of our model is: {train_acc:.3f}')
		print(f'Cross-Validation Test Accuracy of our model is: {test_acc:.3f}')
		print(f'adr validation done in {time.time()-self.start_time:.3f}(s).')
	
	def train(self):
		super().train()
		self.reg = self.reg.fit(self.x_train,self.y_train)
		train_acc = self.reg.score(self.x_train,self.y_train)
		print(f'Training Accuracy of our model is: {train_acc:.3f}')
		print(f'adr training done in {time.time()-self.start_time:.3f}(s).')
		
	def predict(self):
		super().predict()
		predicts = pd.DataFrame(self.reg.predict(self.x_test), columns = ['adr'])
		print(f'adr prediction done in {time.time()-self.start_time:.3f}(s).')
		return predicts

# from feature_engineering import *
# from dataset import Dataset
# hotel = Dataset()
#room_feature = gen_room_feature(hotel.get_feature(['reserved_room_type', 'assigned_room_type']))
#net_canceled_feature = gen_net_canceled_feature(hotel.get_feature(['previous_cancellations', 'previous_bookings_not_canceled']))
#hotel.add_feature(room_feature)
#hotel.add_feature(net_canceled_feature)
#hotel.remove_feature(['children', 'stays_in_weekend_nights'])

# x_train = hotel.get_train_dataset()
# x_test = hotel.get_test_dataset()
# y_train = hotel.get_train_adr()

# lin_reg = TheLinearRegression(x_train, y_train, x_test)
# print(lin_reg.v_fold_validate())
# lin_reg.train()
# print(lin_reg.predict())

# DR = TheDecisionTreeRegressor(x_train, y_train, x_test)
# print(DR.v_fold_validate())
# DR.train()
# print(DR.predict())