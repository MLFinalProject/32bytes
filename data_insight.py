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
        plt.savefig("./data_adr_analysis/img/{}_adr.png".format(attribute))
        # plt.show()

    def plot_attribute_adr_mean(self, attribute):
        attribute_element = self.dataset.get_train_column(attribute).to_numpy().squeeze()
        attribute_element = list(set(attribute_element))
        
        if attribute == "arrival_date_day_of_month":
            attribute_element = [int(i) for i in attribute_element]
            attribute_element.sort()
            attribute_element = [str(i) for i in attribute_element]
            # print(attribute_element)
            # print(type(attribute_element[0]))
        else:
            attribute_element.sort()

        avg_list = []
        num_list = []
        std_list = []

        for i in attribute_element:
            avg = np.average(self.train_df.loc[self.train_df[attribute] == i].loc[:, 'adr'].to_numpy())
            std = np.std(self.train_df.loc[self.train_df[attribute] == i].loc[:, 'adr'].to_numpy())
            num = np.average(len(self.train_df.loc[self.train_df[attribute] == i].loc[:, 'adr'].to_numpy()))
            avg_list.append(avg)
            num_list.append(num)
            std_list.append(std)
            # print("average of {} = {}, which has {} samples".format(i, avg, num))
        # print(len(avg_list))
        # print(len(std_list))
        plt.figure()
        plt.errorbar(attribute_element, avg_list, std_list, linestyle='None', marker='^')
        plt.xlabel(attribute)
        plt.ylabel('adr_w_std')
        # plt.savefig("./data_adr_analysis/img/{}_adr_mean.png".format(attribute))
        plt.savefig("./data_adr_analysis/img_w_std/{}_adr_mean_std.png".format(attribute))
        # plt.show()

        # plt.figure()
        # plt.plot(attribute_element, num_list, 'b.')
        # plt.xlabel(attribute)
        # plt.ylabel('num')
        # # plt.savefig("./data_num_analysis/img/{}_num.png".format(attribute))
        # plt.show()
        # # plt.show()

        data_num_df = pd.DataFrame({attribute: num_list})
        data_num_df.to_csv(r'./data_num_analysis/csv/{}_num.csv'.format(attribute), index=True)

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
        plt.savefig("./data_is_canceled_analysis/img/{}_is_canceled_mean.png".format(attribute))

    def compare_train_test(self, attribute):
        train_attribute = self.dataset.get_train_column(attribute).to_numpy().squeeze()
        test_attribute = self.dataset.get_test_column(attribute).to_numpy().squeeze()
        train_attribute_set, test_attribute_set = set(train_attribute), set(test_attribute)

        train_unique_set = train_attribute_set - test_attribute_set
        return train_unique_set

    def plot_train_unique(self, attribute):
        train_unique_set = self.compare_train_test(attribute)
        train_unique_list = list(train_unique_set)
        train_unique_list.sort()

        avg_is_canceled_list = []
        avg_adr_list = []
        num_list = []

        for i in train_unique_list:
            avg_is_canceled = np.average(self.train_df.loc[self.train_df[attribute] == i].loc[:, 'is_canceled'].to_numpy())
            avg_adr = np.average(self.train_df.loc[self.train_df[attribute] == i].loc[:, 'adr'].to_numpy())
            num = np.average(len(self.train_df.loc[self.train_df[attribute] == i].loc[:, 'is_canceled'].to_numpy()))
            avg_is_canceled_list.append(avg_is_canceled)
            avg_adr_list.append(avg_adr)
            num_list.append(num)

        # plot is canceled
        plt.figure()
        plt.plot(train_unique_list, avg_is_canceled_list, 'r.')
        plt.savefig("./train_unique_is_canceled_analysis/{}_is_canceled_mean.png".format(attribute))

        # plot adr
        plt.figure()
        plt.plot(train_unique_list, avg_adr_list, 'r.')
        plt.savefig("./train_unique_adr_analysis/{}_adr_mean.png".format(attribute))

        # plot num
        plt.figure()
        plt.plot(train_unique_list, num_list, 'r.')
        plt.savefig("./train_unique_adr_analysis/{}_num.png".format(attribute))

    def plot_test(self, attribute):
        test_attribute_list = list(set(self.dataset.get_test_column(attribute).to_numpy().squeeze()))
        test_attribute_list.sort()

        avg_is_canceled_list = []
        avg_adr_list = []
        num_list = []

        for i in test_attribute_list:
            avg_is_canceled = np.average(self.train_df.loc[self.train_df[attribute] == i].loc[:, 'is_canceled'].to_numpy())
            avg_adr = np.average(self.train_df.loc[self.train_df[attribute] == i].loc[:, 'adr'].to_numpy())
            num = np.average(len(self.train_df.loc[self.train_df[attribute] == i].loc[:, 'is_canceled'].to_numpy()))
            avg_is_canceled_list.append(avg_is_canceled)
            avg_adr_list.append(avg_adr)
            num_list.append(num)

        # plot is canceled
        plt.figure()
        plt.plot(test_attribute_list, avg_is_canceled_list, 'r.')
        plt.savefig("./test_is_canceled_analysis/{}_is_canceled_mean.png".format(attribute))

        # plot adr
        plt.figure()
        plt.plot(test_attribute_list, avg_adr_list, 'r.')
        plt.savefig("./test_adr_analysis/{}_adr_mean.png".format(attribute))

        # plot num
        plt.figure()
        plt.plot(test_attribute_list, num_list, 'r.')
        plt.savefig("./test_adr_analysis/{}_num.png".format(attribute))