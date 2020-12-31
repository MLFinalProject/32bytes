from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from dataset import Dataset

class Classification:
    """docstring for ClassName"""
    def __init__(self, x_train, y_train, x_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        

    def v_fold_validate(self):
        self.x_val_train, self.x_val_test, self.y_val_train, self.y_val_test = train_test_split(
            self.x_train, self.y_train, test_size=0.2, random_state=0)


    def train(self):
        print()

    def predict(self):
        print()

class DecisionTree(Classification):
    """docstring for DecisionTree"""
    def __init__(self, x_train, y_train, x_test):
        super().__init__(x_train, y_train, x_test)
        self.dt = DecisionTreeClassifier(random_state = 0)

    def train(self):
        super().train()
        self.dt = self.dt.fit(self.x_train,self.y_train)

    def v_fold_validate(self):
        super().v_fold_validate()
        self.dt = self.dt.fit(self.x_val_train, self.y_val_train)
        train_acc = self.dt.score(self.x_val_train, self.y_val_train)
        test_acc = self.dt.score(self.x_val_test, self.y_val_test)
        print(f'Training Accuracy of our model is: {train_acc}')
        print(f'Test Accuracy of our model is: {test_acc}')

# hotel = Dataset()
# x_train = hotel.get_train_dataset()
# y_train = hotel.get_is_canceled()
# decisiontree = DecisionTree(x_train, y_train, x_train)
# decisiontree.v_fold_validate()
		
		