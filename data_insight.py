import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import math

from dataset import Dataset

class DataInsight():
    def __init__(self):
        self.dataset = Dataset()
        self.train_df = self.dataset.train_test_df[self.dataset.train_test_df['dataset'].eq(self.dataset.data_label["train"])].drop(['dataset'], axis=1)

    def plot_attribute_adr(self, attribute):
        attribute = self.dataset.get_train_column(attribute)
        print(attribute)
        adr = self.dataset.get_train_adr()
        print(adr)
        plt.plot(attribute, adr, ',')
        plt.show()

    def plot_attribute_adr_mean(self, attribute):
        attrbute = self.dataset.get_train_column(attribute)
        attribute = list(set(attribute))
        attribute.sort()
        avg_list = []
        num_list = []
        for i in attribute:
            avg = np.average(self.train_df.loc[self.train_df[attribute] == i].loc[:, 'adr'].to_numpy())
            num = np.average(len(self.train_df.loc[self.train_df['attribute'] == i].loc[:, 'adr'].to_numpy()))
            avg_list.append(avg)
            num_list.append(num)
            print("average of {} = {}, which has {} samples".format(i, avg, num))

        plt.plot(attribute, avg_list, 'r.')
        # plt.plot(attribute, num_list, 'b.')
        plt.show()

    def plot_attribute_is_canceled(self):
        pass