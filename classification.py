import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
import time

class Classification:
    """docstring for ClassName"""
    def __init__(self, x_train, y_train, x_test):
        self.x_train = x_train
        self.y_train = np.ravel(y_train)
        self.x_test = x_test
        

    def v_fold_validate(self):
        print('\n---is_canceled Validating---')
        self.start_time = time.time()
        self.x_val_train, self.x_val_test, self.y_val_train, self.y_val_test = train_test_split(
            self.x_train, self.y_train, test_size=0.2, random_state=0)


    def train(self):
        print('\n---is_canceled Training---')
        self.start_time = time.time()

    def predict(self):
        print('\n---is_canceled Predicting---')
        self.start_time = time.time()

# class TheDecisionTree(Classification):
#     """docstring for DecisionTree"""
#     def __init__(self, x_train, y_train, x_test):
#         super().__init__(x_train, y_train, x_test)
#         self.clf = DecisionTreeClassifier(random_state = 0)

#     def train(self):
#         super().train()
#         self.clf = self.clf.fit(self.x_train,self.y_train)
#         train_acc = self.clf.score(self.x_train, self.y_train)
#         print(f'Training Accuracy of our model is: {train_acc:.3f}')
#         print(f'is_canceled training done in {time.time()-self.start_time:.3f}(s).')

#     def v_fold_validate(self):
#         super().v_fold_validate()
#         self.clf = self.clf.fit(self.x_val_train, self.y_val_train)
#         train_acc = self.clf.score(self.x_val_train, self.y_val_train)
#         test_acc = self.clf.score(self.x_val_test, self.y_val_test)
#         print(f'Training Accuracy of our model is: {train_acc:.3f}')
#         print(f'Test Accuracy of our model is: {test_acc:.3f}')
#         print(f'is_canceled validation done in {time.time()-self.start_time:.3f}(s).')

#     def predict(self):
#         super().predict()
#         predicts = pd.DataFrame(self.clf.predict(self.x_test), columns = ['is_canceled'])
#         print(f'is_canceled prediction done in {time.time()-self.start_time:.3f}(s).')
#         return predicts

# class TheLogisticRegression(Classification):
#     """docstring for DecisionTree"""
#     def __init__(self, x_train, y_train, x_test):
#         super().__init__(x_train, y_train, x_test)
#         self.reg = LogisticRegression(max_iter=800)

#     def train(self):
#         super().train()
#         self.reg = self.reg.fit(self.x_train,self.y_train)
#         train_acc = self.reg.score(self.x_train,self.y_train)
#         print(f'Training Accuracy of our model is: {train_acc:.3f}')
#         print(f'is_canceled training done in {time.time()-self.start_time:.3f}(s).')

#     def predict(self):
#         super().predict()
#         predicts = pd.DataFrame(self.reg.predict(self.x_test), columns = ['is_canceled'])
#         print(f'is_canceled prediction done in {time.time()-self.start_time:.3f}(s).')
#         return predicts

#     def v_fold_validate(self):
#         super().v_fold_validate()
#         self.clf = self.clf.fit(self.x_val_train, self.y_val_train)
#         train_acc = self.clf.score(self.x_val_train, self.y_val_train)
#         test_acc = self.clf.score(self.x_val_test, self.y_val_test)
#         print(f'Training Accuracy of our model is: {train_acc}')
#         print(f'Test Accuracy of our model is: {test_acc}')
#         print(f'is_canceled validation done in {time.time()-self.start_time:.3f}(s).')

class TheRandomForest(Classification):
    """docstring for TheRandomForest"""
    def __init__(self, x_train, y_train, x_test):
        super().__init__(x_train, y_train, x_test)
        self.clf = RandomForestClassifier(min_impurity_decrease=1e-6, n_estimators=128, random_state = 6174, n_jobs = -1)

    def train(self):
        super().train()
        self.clf = self.clf.fit(self.x_train,self.y_train)
        train_acc = self.clf.score(self.x_train, self.y_train)
        print(f'Training Accuracy of our model is: {train_acc:.3f}')
        print(f'is_canceled training done in {time.time()-self.start_time:.3f}(s).')


    def v_fold_validate(self):
        super().v_fold_validate()
        self.clf = self.clf.fit(self.x_val_train, self.y_val_train)
        train_acc = self.clf.score(self.x_val_train, self.y_val_train)
        test_acc = self.clf.score(self.x_val_test, self.y_val_test)
        print(f'Training Accuracy of our model is: {train_acc:.3f}')
        print(f'Test Accuracy of our model is: {test_acc:.3f}')
        print(f'is_canceled validation done in {time.time()-self.start_time:.3f}(s).')

    def predict(self):
        super().predict()
        predicts = pd.DataFrame(self.clf.predict(self.x_test), columns = ['is_canceled'])
        print(f'is_canceled prediction done in {time.time()-self.start_time:.3f}(s).')
        return predicts

    def ensemble(self):
        self.x_val_train, self.x_val_test, self.y_val_train, self.y_val_test = train_test_split(
            self.x_train, self.y_train, test_size=0.2, random_state=1126)
    
    def three_seed_validate(self):
        self.start_time = time.time()
        seed = 1126
        for seed in [123, 1126, 390625]:
            self.x_val_train, self.x_val_test, self.y_val_train, self.y_val_test = train_test_split(
                self.x_train, self.y_train, test_size=0.2, random_state=seed)
            self.clf = self.clf.fit(self.x_val_train, self.y_val_train)
            train_acc = self.clf.score(self.x_val_train, self.y_val_train)
            test_acc = self.clf.score(self.x_val_test, self.y_val_test)
            print(f'seed {seed}\t train_acc:{train_acc:.3f}, test_acc:{test_acc:.3f}')
        print(f'experiment done in {time.time()-self.start_time:.3f}(s).')
        print('--------------------\n')

    def ensemble_seed(self, seed):
        self.start_time = time.time()
        self.clf = RandomForestClassifier(min_impurity_decrease=1e-6, n_estimators=128, random_state = seed, n_jobs = -1)
        self.clf = self.clf.fit(self.x_train,self.y_train)
        train_acc = self.clf.score(self.x_train,self.y_train)
        predicts = pd.DataFrame(self.clf.predict(self.x_test), columns = ['is_canceled'])
        print(f'Classification Accuracy: {train_acc:.3f}', end = '\t')
        print(f'done in {time.time()-self.start_time:.3f}(s).')
        return predicts





# class TheGradientBoost(Classification):
#     """docstring for TheRandomForest"""
#     def __init__(self, x_train, y_train, x_test):
#         super().__init__(x_train, y_train, x_test)
#         self.clf = GradientBoostingClassifier(random_state = 0)

#     def train(self):
#         super().train()
#         self.clf = self.clf.fit(self.x_train,self.y_train)
#         train_acc = self.clf.score(self.x_train, self.y_train)
#         print(f'Training Accuracy of our model is: {train_acc:.3f}')
#         print(f'is_canceled training done in {time.time()-self.start_time:.3f}(s).')


#     def v_fold_validate(self):
#         super().v_fold_validate()
#         self.clf = self.clf.fit(self.x_val_train, self.y_val_train)
#         train_acc = self.clf.score(self.x_val_train, self.y_val_train)
#         test_acc = self.clf.score(self.x_val_test, self.y_val_test)
#         print(f'Training Accuracy of our model is: {train_acc:.3f}')
#         print(f'Test Accuracy of our model is: {test_acc:.3f}')
#         print(f'is_canceled validation done in {time.time()-self.start_time:.3f}(s).')

#     def predict(self):
#         super().predict()
#         predicts = pd.DataFrame(self.clf.predict(self.x_test), columns = ['is_canceled'])
#         print(f'is_canceled prediction done in {time.time()-self.start_time:.3f}(s).')
#         return predicts

# class TheXGBoost(Classification):
#     """docstring for TheRandomForest"""
#     def __init__(self, x_train, y_train, x_test):
#         super().__init__(x_train, y_train, x_test)
#         self.clf = XGBClassifier(seed = 0, use_label_encoder=False)

#     def train(self):
#         super().train()
#         self.clf = self.clf.fit(self.x_train,self.y_train)
#         y_pred = self.clf.predict(self.y_train)
#         print(f'Training Accuracy of our model is: {train_acc:.3f}')
#         print(f'is_canceled training done in {time.time()-self.start_time:.3f}(s).')

#     def v_fold_validate(self):
#         super().v_fold_validate()
#         self.clf = self.clf.fit(self.x_val_train, self.y_val_train, eval_metric='error')
#         train_acc = self.clf.score(self.x_val_train, self.y_val_train)
#         test_acc = self.clf.score(self.x_val_test, self.y_val_test)
#         print(f'Training Accuracy of our model is: {train_acc:.3f}')
#         print(f'Test Accuracy of our model is: {test_acc:.3f}')
#         print(f'is_canceled validation done in {time.time()-self.start_time:.3f}(s).')

#     def predict(self):
#         super().predict()
#         predicts = pd.DataFrame(self.clf.predict(self.x_test), columns = ['is_canceled'])
#         print(f'is_canceled prediction done in {time.time()-self.start_time:.3f}(s).')
#         return predicts        

if __name__ == '__main__':
    from feature_engineering import *
    from dataset import Dataset

    hotel_is_cancel = Dataset()

    # attribute_threshold_dict = {"adults":3, "babies":2, "children":3, "required_car_parking_spaces":2, "stays_in_week_nights":20, "stays_in_weekend_nights":10}
    # for key in attribute_threshold_dict:
    #     new_attribute_df = transfer_not_enough_data_to_mean(hotel_is_cancel.get_feature([key]), attribute_threshold_dict[key])
    #     hotel_is_cancel.remove_feature([key])
    #     hotel_is_cancel.add_feature(new_attribute_df)

    room_feature = gen_room_feature(hotel_is_cancel.get_feature(['reserved_room_type', 'assigned_room_type']))
    net_canceled_feature = gen_net_canceled_feature(hotel_is_cancel.get_feature(['previous_cancellations', 'previous_bookings_not_canceled']))
    hotel_is_cancel.add_feature(room_feature)
    hotel_is_cancel.add_feature(net_canceled_feature)
    x_train_is_canceled = hotel_is_cancel.get_train_dataset()
    x_test_is_canceled = hotel_is_cancel.get_test_dataset()
    y_train_is_canceled = hotel_is_cancel.get_train_is_canceled()
    clf = TheRandomForest(x_train_is_canceled, y_train_is_canceled, x_test_is_canceled)

    predictions = []
    ensemble_count = 0
    clf.ensemble()
    for max_samples_i in [None]:
        for n_estimators_i in [128]:
            for max_depth_i in [None]:
                for random_state_i in [6174]:
                    ensemble_count += 1
                    print(f'No.{ensemble_count} experiment n_estimators = {n_estimators_i}, max_depth = {max_depth_i}, max_samples = {max_samples_i}, seed = {random_state_i}.')
                    clf.clf = RandomForestClassifier(min_impurity_decrease=1e-6,n_estimators=n_estimators_i, n_jobs = -1, max_depth = max_depth_i, bootstrap=True, max_samples = max_samples_i, random_state = random_state_i)                    
                    clf.three_seed_validate()
                    # predictions.append(clf.clf.predict(clf.x_val_test))
                    # 0.4, 512, 50
    # predictions = np.stack(predictions)
    # y_pred = np.sum(predictions, axis=0)
    # threshold = ensemble_count//2
    # print(f'ensemble_count = {ensemble_count}, threshold = {threshold}')
    # y_pred[y_pred <= threshold] = 0
    # y_pred[y_pred > threshold] = 1
    # ensemble_acc = np.sum(clf.y_val_test == y_pred)/len(y_pred)
    # print(f'ensemble_acc is {ensemble_acc:.3f}')
		
		