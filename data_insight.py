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
        self.train_df['adr'] = self.dataset.target_df['adr']
        self.train_df['is_canceled'] = self.dataset.target_df['is_canceled']

    def plot_attribute_adr(self, attribute):
        attribute_element = self.dataset.get_train_column(attribute).to_numpy().squeeze()
        # print(attribute)
        adr = self.dataset.get_train_adr().to_numpy().squeeze()
        # print(adr)
        plt.figure()
        plt.plot(attribute_element, adr, ',')
        plt.savefig("./data_adr_analysis/{}_adr.png".format(attribute))
        # plt.show()

    def plot_attribute_adr_mean(self, attribute):
        attribute_element = self.dataset.get_train_column(attribute).to_numpy().squeeze()
        attribute_element = list(set(attribute_element))
        attribute_element.sort()
        avg_list = []
        num_list = []
        for i in attribute_element:
            avg = np.average(self.train_df.loc[self.train_df[attribute] == i].loc[:, 'adr'].to_numpy())
            num = np.average(len(self.train_df.loc[self.train_df[attribute] == i].loc[:, 'adr'].to_numpy()))
            avg_list.append(avg)
            num_list.append(num)
            # print("average of {} = {}, which has {} samples".format(i, avg, num))
        plt.figure()
        plt.plot(attribute_element, avg_list, 'r.')
        plt.savefig("./data_adr_analysis/{}_adr_mean.png".format(attribute))
        # plt.show()

    def plot_attribute_is_canceled_mean(self, attribute):
        attribute_element = self.dataset.get_train_column(attribute).to_numpy().squeeze()
        attribute_element = list(set(attribute_element))
        attribute_element.sort()
        avg_list = []
        num_list = []
        for i in attribute_element:
            avg = np.average(self.train_df.loc[self.train_df[attribute] == i].loc[:, 'is_canceled'].to_numpy())
            num = np.average(len(self.train_df.loc[self.train_df[attribute] == i].loc[:, 'is_canceled'].to_numpy()))
            avg_list.append(avg)
            num_list.append(num)
            # print("average of {} = {}, which has {} samples".format(i, avg, num))
        plt.figure()
        plt.plot(attribute_element, avg_list, 'r.')
        plt.savefig("./data_is_canceled_analysis/{}_is_canceled_mean.png".format(attribute))