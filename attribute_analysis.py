import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import math

month_str2num = {'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6,
                 'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12}

train_df = pd.read_csv('./data/train.csv')
train_label_np = pd.read_csv('./data/train_label.csv').to_numpy()
arrival_date = train_df.loc[:,['arrival_date_year','arrival_date_month','arrival_date_day_of_month']].to_numpy()

arrival_date[:,1] = [month_str2num[i] for i in arrival_date[:,1]]
arrival_date = [date(i[0],i[1],i[2]).isoformat() for i in arrival_date]

number_of_days = train_df.loc[:,['stays_in_weekend_nights', 'stays_in_week_nights']].to_numpy()
number_of_days = np.sum(number_of_days, axis=1)

adr = train_df.loc[:,'adr'].to_numpy()

# # --- hotel ---
# hotel = train_df.loc[:,'hotel'].to_numpy()
# plt.plot(hotel, adr, ',')
# plt.show()

# # average for hotel
# df_city_hotel = train_df.loc[train_df['hotel'] == 'City Hotel']
# df_resort_hotel = train_df.loc[train_df['hotel'] == 'Resort Hotel']
# adr_city_hotel = df_city_hotel.loc[:, 'adr'].to_numpy()
# adr_resort_hotel = df_resort_hotel.loc[:, 'adr'].to_numpy()
# adr_avg_city_hotel = np.average(adr_city_hotel)
# adr_avg_resort_hotel = np.average(adr_resort_hotel)
# print('adr_avg_city_hotel = ', adr_avg_city_hotel, '\t', 'adr_avg_resort_hotel = ', adr_avg_resort_hotel)

# # --- is canceled ---
# is_canceled = train_df.loc[:,'is_canceled']

# plt.plot(adr, is_canceled, ',')
# plt.axis([290, 300, -0.1, 1.1])
# plt.show()

# # The ploting result is too hard to recognize by human eyes, I calculated the average of adr for both the cases of whether it's canceled

# adr_avg_is_canceled = np.average(train_df.loc[train_df['is_canceled'] == 1].loc[:, 'adr'].to_numpy())
# adr_avg_not_canceled = np.average(train_df.loc[train_df['is_canceled'] == 0].loc[:, 'adr'].to_numpy())
# print('adr_avg_is_canceled = ', adr_avg_is_canceled, '\t', 'adr_avg_not_canceled = ', adr_avg_not_canceled)


# # --- lead_time ---
# lead_time = train_df.loc[:, 'lead_time']
# plt.plot(lead_time, adr, ',')
# plt.ylim(0, 400)
# plt.show()
# # It seems the smaller the lead time, the higer the adr

# #  --- arrival_date_year ---
# arrival_date_year = train_df.loc[:, 'arrival_date_year']
# plt.plot(arrival_date_year, adr, ',')
# plt.ylim(0, 400)
# plt.show()

# arrival_date_year = set(arrival_date_year)
# for i in arrival_date_year:
#     avg = np.average(train_df.loc[train_df['arrival_date_year'] == i].loc[:, 'adr'].to_numpy())
#     print("average of {} = {}".format(i, avg))


# #  --- arrival_date_month ---
# arrival_date_month = train_df.loc[:, 'arrival_date_month']
# plt.plot(arrival_date_month, adr, ',')
# plt.ylim(0, 400)
# plt.show()

# arrival_date_month = set(arrival_date_month)
# for i in arrival_date_month:
#     avg = np.average(train_df.loc[train_df['arrival_date_month'] == i].loc[:, 'adr'].to_numpy())
#     print("average of {} = {}".format(i, avg))


# #  --- arrival_date_week_number ---
# arrival_date_week_number = train_df.loc[:, 'arrival_date_week_number']
# plt.plot(arrival_date_week_number, adr, ',')
# # plt.ylim(0, 400)
# plt.show()

# arrival_date_week_number = set(arrival_date_week_number)
# for i in arrival_date_week_number:
#     avg = np.average(train_df.loc[train_df['arrival_date_week_number'] == i].loc[:, 'adr'].to_numpy())
#     print("average of {} = {}".format(i, avg))



# #  --- arrival_date_day_of_month ---
# arrival_date_day_of_month = train_df.loc[:, 'arrival_date_day_of_month']
# plt.plot(arrival_date_day_of_month, adr, ',')
# # plt.ylim(0, 400)
# plt.show()

# arrival_date_day_of_month = set(arrival_date_day_of_month)
# for i in arrival_date_day_of_month:
#     avg = np.average(train_df.loc[train_df['arrival_date_day_of_month'] == i].loc[:, 'adr'].to_numpy())
#     print("average of {} = {}".format(i, avg))


# #  --- stays_in_weekend_nights ---
# stays_in_weekend_nights = train_df.loc[:, 'stays_in_weekend_nights']
# plt.plot(stays_in_weekend_nights, adr, ',')
# plt.ylim(0, 1000)
# plt.show()

# stays_in_weekend_nights = set(stays_in_weekend_nights)
# for i in stays_in_weekend_nights:
#     avg = np.average(train_df.loc[train_df['stays_in_weekend_nights'] == i].loc[:, 'adr'].to_numpy())
#     print("average of {} = {}".format(i, avg))


# #  --- stays_in_week_nights ---
# stays_in_week_nights = train_df.loc[:, 'stays_in_week_nights']
# plt.plot(stays_in_week_nights, adr, ',')
# plt.ylim(0, 1000)
# plt.show()

# stays_in_week_nights = list(set(stays_in_week_nights))
# stays_in_week_nights.sort()
# for i in stays_in_week_nights:
#     avg = np.average(train_df.loc[train_df['stays_in_week_nights'] == i].loc[:, 'adr'].to_numpy())
#     print("average of {} = {}".format(i, avg))


# #  --- adults ---
# adults = train_df.loc[:, 'adults']
# plt.plot(adults, adr, ',')
# plt.ylim(0, 1000)
# plt.show()

# adults = list(set(adults))
# adults.sort()
# for i in adults:
#     avg = np.average(train_df.loc[train_df['adults'] == i].loc[:, 'adr'].to_numpy())
#     print("average of {} = {}".format(i, avg))


# #  --- children ---
# children = train_df.loc[:, 'children']
# plt.plot(children, adr, ',')
# plt.ylim(0, 1000)
# plt.show()

# children = list(set(children))
# children.sort()
# for i in children:
#     if not math.isnan(i): # Warning: I remove nan data
#         avg = np.average(train_df.loc[train_df['children'] == i].loc[:, 'adr'].to_numpy())
#         print("average of {} = {}".format(i, avg))


# #  --- babies ---
# babies = train_df.loc[:, 'babies']
# plt.plot(babies, adr, ',')
# plt.ylim(0, 1000)
# plt.show()

# babies = list(set(babies))
# babies.sort()
# for i in babies:
#     avg = np.average(train_df.loc[train_df['babies'] == i].loc[:, 'adr'].to_numpy())
#     print("average of {} = {}".format(i, avg))



# #  --- meal ---
# meal = train_df.loc[:, 'meal']
# plt.plot(meal, adr, ',')
# plt.ylim(0, 1000)
# plt.show()

# meal = list(set(meal))
# meal.sort()
# for i in meal:
#     avg = np.average(train_df.loc[train_df['meal'] == i].loc[:, 'adr'].to_numpy())
#     print("average of {} = {}".format(i, avg))


#  --- country ---
new_train_df = train_df.loc[:, ['country', 'adr']].dropna()
country = new_train_df.loc[:, 'country']
adr_country = new_train_df.loc[:, 'adr']
plt.plot(country, adr_country, ',')
plt.ylim(0, 1000)
plt.show()

country = list(set(country))
print(country)
country.sort()
for i in country:
    avg = np.average(train_df.loc[train_df['country'] == i].loc[:, 'adr'].to_numpy())
    print("average of {} = {}".format(i, avg))




# adr_avg_is_canceled = np.average(train_df.loc[train_df['is_canceled'] == 1].loc[:, 'adr'].to_numpy())
# adr_avg_not_canceled = np.average(train_df.loc[train_df['is_canceled'] == 0].loc[:, 'adr'].to_numpy())




# total_adr = {}
# for i in range(len(is_canceled)):
# 	try:
# 		total_adr[arrival_date[i]] += (-is_canceled[i] + 1)*adr[i]*number_of_days[i]	
# 	except:
# 		total_adr[arrival_date[i]] = (-is_canceled[i] + 1)*adr[i]*number_of_days[i]

# x = []
# y = []
# for i in range(len(train_label_np)):
# 	x.append(train_label_np[i][1])
# 	y.append(total_adr[train_label_np[i][0]])
# plt.plot(x,y, ',')
# plt.xlabel('y_score (range from 1 to 10')
# plt.ylabel('adr_sum')
# plt.title('adr_sum - y_score')
# plt.show()

