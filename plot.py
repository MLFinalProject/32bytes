from data_insight import DataInsight
from dataset import Dataset
import pandas as pd

data_insight = DataInsight()
# dataset = Dataset()
# train_test_df = pd.get_dummies(dataset.train_test_df)
# print(dataset.train_test_df)
# print(dataset.get_train_dataset())

attribute_list = list(data_insight.train_df.columns.values)[:-2]
for attribute in attribute_list:
    # data_insight.plot_attribute_adr(attribute)
    data_insight.plot_attribute_adr_mean(attribute)
    # data_insight.plot_attribute_is_canceled_mean(attribute)


# data_insight.plot_attribute_adr_mean("country")