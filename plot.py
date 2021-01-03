from data_insight import DataInsight
from dataset import Dataset
import pandas as pd

data_insight = DataInsight()
# dataset = Dataset()
# train_test_df = pd.get_dummies(dataset.train_test_df)
# print(dataset.train_test_df)
# print(dataset.get_train_dataset())
data_insight.plot_attribute_adr('country')
