import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
import time

class Classification:
    """docstring for ClassName"""
    def __init__(self, x_train, y_train, x_test):
        self.x_train = x_train
        self.y_train = y_train
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

class TheDecisionTree(Classification):
    """docstring for DecisionTree"""
    def __init__(self, x_train, y_train, x_test):
        super().__init__(x_train, y_train, x_test)
        self.clf = DecisionTreeClassifier(random_state = 0)

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

class TheLogisticRegression(Classification):
    """docstring for DecisionTree"""
    def __init__(self, x_train, y_train, x_test):
        super().__init__(x_train, y_train, x_test)
        self.reg = LogisticRegression(max_iter=800)

    def train(self):
        super().train()
        self.reg = self.reg.fit(self.x_train,self.y_train)
        train_acc = self.reg.score(self.x_train,self.y_train)
        print(f'Training Accuracy of our model is: {train_acc:.3f}')
        print(f'is_canceled training done in {time.time()-self.start_time:.3f}(s).')

    def predict(self):
        super().predict()
        predicts = pd.DataFrame(self.reg.predict(self.x_test), columns = ['is_canceled'])
        print(f'is_canceled prediction done in {time.time()-self.start_time:.3f}(s).')
        return predicts

    def v_fold_validate(self):
        super().v_fold_validate()
        self.clf = self.clf.fit(self.x_val_train, self.y_val_train)
        train_acc = self.clf.score(self.x_val_train, self.y_val_train)
        test_acc = self.clf.score(self.x_val_test, self.y_val_test)
        print(f'Training Accuracy of our model is: {train_acc}')
        print(f'Test Accuracy of our model is: {test_acc}')
        print(f'is_canceled validation done in {time.time()-self.start_time:.3f}(s).')

class TheRandomForest(Classification):
    """docstring for TheRandomForest"""
    def __init__(self, x_train, y_train, x_test):
        super().__init__(x_train, y_train, x_test)
        self.clf = RandomForestClassifier(n_estimators=100, random_state = 0)

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

class TheXGBoost(Classification):
    """docstring for TheRandomForest"""
    def __init__(self, x_train, y_train, x_test):
        super().__init__(x_train, y_train, x_test)
        self.clf = XGBClassifier(seed = 0, use_label_encoder=False)

    def train(self):
        super().train()
        self.clf = self.clf.fit(self.x_train,self.y_train)
        y_pred = self.clf.predict(self.y_train)
        print(f'Training Accuracy of our model is: {train_acc:.3f}')
        print(f'is_canceled training done in {time.time()-self.start_time:.3f}(s).')

    def v_fold_validate(self):
        super().v_fold_validate()
        self.clf = self.clf.fit(self.x_val_train, self.y_val_train, eval_metric='error')
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

# from feature_engineering import *
# from dataset import Dataset
# hotel = Dataset()
# room_feature = gen_room_feature(hotel.get_feature(['reserved_room_type', 'assigned_room_type']))
# net_canceled_feature = gen_net_canceled_feature(hotel.get_feature(['previous_cancellations', 'previous_bookings_not_canceled']))
# hotel.add_feature(room_feature)
# hotel.add_feature(net_canceled_feature)
# hotel.remove_feature(['children', 'stays_in_weekend_nights'])
# x_train = hotel.get_train_dataset()
# x_test = hotel.get_test_dataset()
# y_train = hotel.get_train_is_canceled()
# decisiontree = TheDecisionTree(x_train, y_train, x_test)
# decisiontree.train()
# print(decisiontree.v_fold_validate())

		
		