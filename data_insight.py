import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import math

from dataset import Dataset

class DataInsight():
    def __init__(self):
        self.dataset = Dataset()

    def plot_attribute_adr(self, attribute):
        attribute = self.dataset.get_column(attribute, True)
        adr = self.dataset.get_adr(True)
        plt.plot(attribute, adr, ',')
        plt.show()

    def plot_attribute_adr_mean(self, attribute):
        attrbute = self.dataset.get_column(attribute, True)
        attribute = list(set(attribute))
        attribute.sort()
        avg_list = []
        num_list = []
        for i in attribute:
            avg = np.average(self.dataset.train_df.loc[self.dataset.train_df[attribute] == i].loc[:, 'adr'].to_numpy())
            num = np.average(len(self.dataset.train_df.loc[train_df['attribute'] == i].loc[:, 'adr'].to_numpy()))
            avg_list.append(avg)
            num_list.append(num)
            print("average of {} = {}, which has {} samples".format(i, avg, num))

        plt.plot(attribute, avg_list, 'r.')
        # plt.plot(attribute, num_list, 'b.')
        plt.show()

    def plot_attribute_is_canceled(self):
        pass