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

    def ensemble(self):
        self.x_val_train, self.x_val_test, self.y_val_train, self.y_val_test = train_test_split(
            self.x_train, self.y_train, test_size=0.2, random_state=1126)
    
    def three_seed_validate(self):
        self.start_time = time.time()
        seed = 1126
        # for seed in [123, 1126, 390625]:
        #     self.x_val_train, self.x_val_test, self.y_val_train, self.y_val_test = train_test_split(
        #         self.x_train, self.y_train, test_size=0.2, random_state=seed)
        self.reg = self.reg.fit(self.x_val_train, self.y_val_train)
        train_acc = self.reg.score(self.x_val_train, self.y_val_train)
        test_acc = self.reg.score(self.x_val_test, self.y_val_test)
        print(f'seed {seed}\t train_acc:{train_acc:.3f}, test_acc:{test_acc:.3f}')
        print(f'experiment done in {time.time()-self.start_time:.3f}(s).')
        print('--------------------\n')

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

if __name__ == '__main__':
    from feature_engineering import *
    from dataset import Dataset

    hotel_adr = Dataset()
    x_train_adr = hotel_adr.get_train_dataset()
    x_test_adr = hotel_adr.get_test_dataset()
    y_train_adr = hotel_adr.get_train_adr()
    reg = TheRandomForestRegressor(x_train_adr, y_train_adr, x_test_adr)

    predictions = []
    ensemble_count = 0
    reg.ensemble()
    for max_samples_i in [None]:
        for n_estimators_i in [512]:
            for max_depth_i in [None]:
                for random_state_i in [1126]:
                    ensemble_count += 1
                    print(f'No.{ensemble_count} experiment n_estimators = {n_estimators_i}, max_depth = {max_depth_i}, max_samples = {max_samples_i}, seed = {random_state_i}.')
                    reg.reg = RandomForestRegressor(min_impurity_decrease=0.005, max_features=0.33, min_samples_leaf = 2, n_estimators=n_estimators_i, max_depth = max_depth_i, max_samples = max_samples_i, random_state = random_state_i, bootstrap=True, n_jobs = -1)                    
                    reg.three_seed_validate()
                    # predictions.append(reg.reg.predict(reg.x_val_test))
                    # 0.4, 512, 50
    # predictions = np.stack(predictions)
    # y_pred = np.sum(predictions, axis=0)
    # threshold = ensemble_count//2
    # print(f'ensemble_count = {ensemble_count}, threshold = {threshold}')
    # y_pred[y_pred <= threshold] = 0
    # y_pred[y_pred > threshold] = 1
    # ensemble_acc = np.sum(reg.y_val_test == y_pred)/len(y_pred)
    # print(f'ensemble_acc is {ensemble_acc:.3f}')
