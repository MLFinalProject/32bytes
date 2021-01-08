import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
import time

class Regression:
    def __init__(self, x_train, y_train, x_test):
        self.x_train = x_train
        self.y_train = np.ravel(y_train)
        self.x_test = x_test

    def v_fold_validate(self):
        print('\n---adr validating---')
        self.start_time = time.time()
        self.x_val_train, self.x_val_test, self.y_val_train, self.y_val_test = train_test_split(self.x_train, self.y_train, test_size = 0.2, random_state = 390625)

    def monthly_validate(self, seed):
        print(f'---adr validating each month---')
        self.month_str = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December',]
        self.start_time = time.time()
        self.x_val_train, self.x_val_test, self.y_val_train, self.y_val_test = train_test_split(
            self.x_train, self.y_train, test_size=0.5, random_state=seed)
      
        self.y_val_test = pd.DataFrame(self.y_val_test.tolist(),columns=['y'])
        self.x_val_test = pd.concat([self.x_val_test.reset_index(drop=True), self.y_val_test],axis=1)

        
        self.x_month_test = {}
        self.y_month_test = {}
        for m in self.month_str:
            column_label = f'arrival_date_month_{m}'
            x_val_test = self.x_val_test[self.x_val_test[column_label].eq(1)]
            self.x_month_test[m] = x_val_test.drop(['y'], axis = 1)
            self.y_month_test[m] = x_val_test['y'].to_numpy()
        self.x_val_test.drop(['y'], axis = 1, inplace = True)

    def train(self):
        print('\n---adr training---')
        self.start_time = time.time()

    def predict(self):
        print('\n---adr predicting---')
        self.start_time = time.time()

class TheRandomForestRegressor(Regression):
    def __init__(self, x_train, y_train, x_test):
        super().__init__(x_train, y_train, x_test)
        self.reg = RandomForestRegressor(min_impurity_decrease=0.001, max_features=.55, min_samples_leaf = 2, n_estimators=128, random_state = 6174, n_jobs = -1)

    def v_fold_validate(self):
        super().v_fold_validate()
        self.reg = self.reg.fit(self.x_val_train, self.y_val_train)
        train_err = self.reg.score(self.x_val_train, self.y_val_train)
        test_err = self.reg.score(self.x_val_test, self.y_val_test)
        print(f'Training Accuracy of our model is: {train_err:.3f}')
        print(f'Test Accuracy of our model is: {test_err:.3f}')
        print(f'adr validation done in {time.time()-self.start_time:.3f}(s).')

    def monthly_validate(self, seed = None):
        super().monthly_validate(seed)
        self.reg = self.reg.fit(self.x_val_train, self.y_val_train)
        y_train_pred = self.reg.predict(self.x_val_train)
        y_test_pred = self.reg.predict(self.x_val_test)
        train_err = mean_absolute_error(y_train_pred, self.y_val_train)
        test_err = mean_absolute_error(y_test_pred, self.y_val_test)
        print(f'Overall||\ntrain_err: {train_err:.3f}\ntest_err: {test_err:.3f}')
        print('--------------------\nMonthly||')
        month_acc = []
        for m in self.month_str:
            y_test_pred = self.reg.predict(self.x_month_test[m])
            test_err = mean_absolute_error(y_test_pred, self.y_month_test[m])
            month_acc.append(test_err)
            print(f'test_err: {test_err:.3f} ({m})')
        print(f'mean: {np.mean(month_acc):.3f}, std: {np.std(month_acc):.3f}, max: {np.max(month_acc):.3f}, min: {np.min(month_acc):.3f}')
        print(f'mean: {np.mean(month_acc[3:8]):.3f}, std: {np.std(month_acc[3:8]):.3f}, max: {np.max(month_acc[3:8]):.3f}, min: {np.min(month_acc[3:8]):.3f} (April-August)')
        print(f'done in {time.time()-self.start_time:.3f}(s).\n')

    def ensemble(self):
        self.x_val_train, self.x_val_test, self.y_val_train, self.y_val_test = train_test_split(
            self.x_train, self.y_train, test_size=0.2, random_state=1126)

    def ensemble_seed(self, seed):
        self.start_time = time.time()
        self.reg = RandomForestRegressor(min_impurity_decrease=0.001, max_features=0.4, min_samples_leaf = 2, n_estimators=128, random_state = seed, n_jobs = -1)
        self.reg = self.reg.fit(self.x_train,self.y_train)
        train_err = self.reg.score(self.x_train,self.y_train)
        predicts = pd.DataFrame(self.reg.predict(self.x_test), columns = ['adr'])
        print(f'Regression Accuracy: {train_err:.3f}', end = '\t')
        print(f'done in {time.time()-self.start_time:.3f}(s).')
        return predicts
    
    def three_seed_validate(self):
        self.start_time = time.time()
        seed = 1126
        for seed in [123, 1126, 390625]:
            self.x_val_train, self.x_val_test, self.y_val_train, self.y_val_test = train_test_split(
                self.x_train, self.y_train, test_size=0.2, random_state=seed)
            self.reg = self.reg.fit(self.x_val_train, self.y_val_train)
            train_err = self.reg.score(self.x_val_train, self.y_val_train)
            test_err = self.reg.score(self.x_val_test, self.y_val_test)
            print(f'seed {seed}\t train_err:{train_err:.3f}, test_err:{test_err:.3f}')
        print(f'experiment done in {time.time()-self.start_time:.3f}(s).')
        print('--------------------\n')

    def train(self):
        super().train()
        self.reg = self.reg.fit(self.x_train,self.y_train)
        train_err = self.reg.score(self.x_train,self.y_train)
        print(f'Training Accuracy of our model is: {train_err:.3f}')
        print(f'adr training done in {time.time()-self.start_time:.3f}(s).')
        
    def predict(self):
        super().predict()
        predicts = pd.DataFrame(self.reg.predict(self.x_test), columns = ['adr'])
        print(f'adr prediction done in {time.time()-self.start_time:.3f}(s).')
        return predicts


		
# class TheLinearRegression(Regressi on):
#     def __init__(self, x_train, y_train, x_test):
#         super().__init__(x_train, y_train, x_test)
#         self.reg = LinearRegression()

#     def v_fold_validate(self):
#         super().v_fold_validate()
#         self.reg = self.reg.fit(self.x_val_train, self.y_val_train)
#         train_err = self.reg.score(self.x_val_train, self.y_val_train)
#         test_err = self.reg.score(self.x_val_test, self.y_val_test)
#         print('---Cross-Validation Testing---')
#         print(f'Training Accuracy of our model is: {train_err}')
#         print(f'Cross-Validation Test Accuracy of our model is: {test_err}')
#         print(f'adr validation done in {time.time()-self.start_time:.3f}(s).')
	
#     def train(self):
#         super().train()
#         self.reg = self.reg.fit(self.x_train,self.y_train)
#         train_err = self.reg.score(self.x_train,self.y_train)
#         print(f'Training Accuracy of our model is: {train_err:.3f}')
#         print(f'adr training done in {time.time()-self.start_time:.3f}(s).')

#     def monthly_validate(self, seed = None):
#         super().monthly_validate(seed)
#         self.reg = self.reg.fit(self.x_val_train, self.y_val_train)
#         y_train_pred = self.reg.predict(self.x_val_train)
#         y_test_pred = self.reg.predict(self.x_val_test)
#         train_err = mean_absolute_error(y_train_pred, self.y_val_train)
#         test_err = mean_absolute_error(y_test_pred, self.y_val_test)
#         print(f'Overall||\ntrain_err: {train_err:.3f}\ntest_err: {test_err:.3f}')
#         print('--------------------\nMonthly||')
#         month_acc = []
#         for m in self.month_str:
#             y_test_pred = self.reg.predict(self.x_month_test[m])
#             test_err = mean_absolute_error(y_test_pred, self.y_month_test[m])
#             month_acc.append(test_err)
#             print(f'test_err: {test_err:.3f} ({m})')
    #     print(f'mean: {np.mean(month_acc):.3f}, std: {np.std(month_acc):.3f}, max: {np.max(month_acc):.3f}, min: {np.min(month_acc):.3f}')
    #     print(f'mean: {np.mean(month_acc[3:8]):.3f}, std: {np.std(month_acc[3:8]):.3f}, max: {np.max(month_acc[3:8]):.3f}, min: {np.min(month_acc[3:8]):.3f} (April-August)')
    #     print(f'done in {time.time()-self.start_time:.3f}(s).\n')
    
    # def predict(self):
    #     super().predict()
    #     predicts = pd.DataFrame(self.reg.predict(self.x_test), columns = ['adr'])
    #     print(f'adr prediction done in {time.time()-self.start_time:.3f}(s).')
    #     return predicts

# class TheDecisionTreeRegressor(Regression):
# 	def __init__(self, x_train, y_train, x_test):
# 		super().__init__(x_train, y_train, x_test)
# 		self.reg = DecisionTreeRegressor(random_state = 5)
	
# 	def v_fold_validate(self):
# 		super().v_fold_validate()
# 		self.reg = self.reg.fit(self.x_val_train, self.y_val_train)
# 		train_err = self.reg.score(self.x_val_train, self.y_val_train)
# 		test_err = self.reg.score(self.x_val_test, self.y_val_test)
# 		print('---Cross-Validation Testing---')
# 		print(f'Training Accuracy of our model is: {train_err:.3f}')
# 		print(f'Cross-Validation Test Accuracy of our model is: {test_err:.3f}')
# 		print(f'adr validation done in {time.time()-self.start_time:.3f}(s).')
	
# 	def train(self):
# 		super().train()
# 		self.reg = self.reg.fit(self.x_train,self.y_train)
# 		train_err = self.reg.score(self.x_train,self.y_train)
# 		print(f'Training Accuracy of our model is: {train_err:.3f}')
# 		print(f'adr training done in {time.time()-self.start_time:.3f}(s).')
		
# 	def predict(self):
# 		super().predict()
# 		predicts = pd.DataFrame(self.reg.predict(self.x_test), columns = ['adr'])
# 		print(f'adr prediction done in {time.time()-self.start_time:.3f}(s).')
# 		return predicts

# class TheGradientBoostingRegressor(Regression):
# 	def __init__(self, x_train, y_train, x_test):
# 		super().__init__(x_train, y_train, x_test)
# 		self.reg = GradientBoostingRegressor(random_state = 0, n_estimators = 100, loss='lad', max_depth = 4)
	
# 	def v_fold_validate(self):
# 		super().v_fold_validate()
# 		self.reg = self.reg.fit(self.x_val_train, self.y_val_train)
# 		train_err = self.reg.score(self.x_val_train, self.y_val_train)
# 		test_err = self.reg.score(self.x_val_test, self.y_val_test)
# 		print('---Cross-Validation Testing---')
# 		print(f'Training Accuracy of our model is: {train_err:.3f}')
# 		print(f'Cross-Validation Test Accuracy of our model is: {test_err:.3f}')
# 		print(f'adr validation done in {time.time()-self.start_time:.3f}(s).')
	
# 	def train(self):
# 		super().train()
# 		self.reg = self.reg.fit(self.x_train,self.y_train)
# 		train_err = self.reg.score(self.x_train,self.y_train)
# 		print(f'Training Accuracy of our model is: {train_err:.3f}')
# 		print(f'adr training done in {time.time()-self.start_time:.3f}(s).')
		
# 	def predict(self):
# 		super().predict()
# 		predicts = pd.DataFrame(self.reg.predict(self.x_test), columns = ['adr'])
# 		print(f'adr prediction done in {time.time()-self.start_time:.3f}(s).')
# 		return predicts



if __name__ == '__main__':
    from feature_engineering import *
    from dataset import Dataset
    hotel_adr = Dataset()
    # attribute_threshold_dict = {"adults":3, "babies":2, "children":3, "required_car_parking_spaces":2, "stays_in_week_nights":20, "stays_in_weekend_nights":10}
    # for key in attribute_threshold_dict:
    #     new_attribute_df = transfer_not_enough_data_to_mean(hotel_adr.get_feature([key]), attribute_threshold_dict[key])
    #     hotel_adr.remove_feature([key])
    #     hotel_adr.add_feature(new_attribute_df)
    # modified_key = "arrival_date_week_number"
    # peak = 34
    # new_attribute_df = absolute_peak_transform(hotel_adr.get_feature([modified_key]), peak)
    # hotel_adr.remove_feature([modified_key])
    # hotel_adr.add_feature(new_attribute_df)
    # hotel_adr.train_test_df['arrival_date_week_number'] = hotel_adr.train_test_df['arrival_date_week_number'].apply(str)
    # hotel_adr.remove_feature(['agent','company'])

    # room_feature = gen_room_feature(hotel_adr.get_feature(['reserved_room_type', 'assigned_room_type']))
    # net_canceled_feature = gen_net_canceled_feature(hotel_adr.get_feature(['previous_cancellations', 'previous_bookings_not_canceled']))
    # hotel_adr.add_feature(room_feature)
    # hotel_adr.add_feature(net_canceled_feature)

    # remove_only_list = ['country', 'agent', 'company']
    # for only_attribute in remove_only_list:
    #     attribute_train_column = hotel_adr.get_train_column(only_attribute)
    #     attribute_test_column = hotel_adr.get_test_column(only_attribute)
    #     new_attribute_column = remove_only(hotel_adr.get_feature([only_attribute]), attribute_train_column, attribute_test_column)
    #     hotel_adr.remove_feature([only_attribute])
    #     hotel_adr.add_feature(new_attribute_column)

    x_train_adr = hotel_adr.get_train_dataset()
    x_test_adr = hotel_adr.get_test_dataset()
    y_train_adr = hotel_adr.get_train_adr()

    # for only_attribute in remove_only_list:
    #     remove_string = '{}_RMV'.format(only_attribute)
    #     x_train_adr.drop([remove_string], axis=1, inplace=True)
    #     x_test_adr.drop([remove_string], axis=1, inplace=True)

    reg = TheRandomForestRegressor(x_train_adr, y_train_adr, x_test_adr)
    reg.reg = RandomForestRegressor(min_impurity_decrease=0.001, max_features=.55, min_samples_leaf = 2, n_estimators=128, random_state = 6174, bootstrap=True, n_jobs = -1)
    reg.monthly_validate(123)
    reg.monthly_validate(1126)
    reg.monthly_validate(390625)

    # reg.three_seed_validate()
    # exit()

    # predictions = []
    # ensemble_count = 0
    # # reg.ensemble()
    # for min_weight_fraction_leaf_i in [0.0]:
    #     for min_impurity_decrease_i in [0.001]:
    #         for max_features_i in [0.4]:
    #             for min_samples_leaf_i in [2]:
    #                 ensemble_count += 1
    #                 print(f'No.{ensemble_count} experiment min_weight_fraction_leaf = {min_weight_fraction_leaf_i}, min_impurity_decrease = {min_impurity_decrease_i}, max_features = {max_features_i}, min_samples_leaf = {min_samples_leaf_i}.')
    #                 reg.reg = RandomForestRegressor(min_weight_fraction_leaf=min_weight_fraction_leaf_i, min_impurity_decrease=min_impurity_decrease_i, max_features=max_features_i, min_samples_leaf = min_samples_leaf_i, n_estimators=128, max_depth = None, max_samples = None, random_state = 1126, bootstrap=True, n_jobs = -1)                    
    #                 reg.three_seed_validate()
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
