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
avg_list = []
num_list = []
for i in country:
    avg = np.average(train_df.loc[train_df['country'] == i].loc[:, 'adr'].to_numpy())
    num = np.average(len(train_df.loc[train_df['country'] == i].loc[:, 'adr'].to_numpy()))
    avg_list.append(avg)
    num_list.append(num)
    print("average of {} = {}, which has {} samples".format(i, avg, num))

# The code below are wrong due to its sorting method
avg_list = [avg for _, avg in sorted(zip(num_list, avg_list))]
country = [cou for _, cou in sorted(zip(num_list, country))]
num_list.sort()

print(country)
print(num_list)
print(avg_list)

print(max(avg_list))
print(max(num_list))
plt.plot(country, avg_list, 'r.')
# plt.plot(country, num_list, 'b.')
plt.show()



# #  --- market_segment ---
# market_segment = train_df.loc[:, 'market_segment']
# plt.plot(market_segment, adr, ',')
# plt.ylim(0, 1000)
# plt.show()

# market_segment = list(set(market_segment))
# market_segment.sort()
# avg_list = []
# num_list = []
# for i in market_segment:
#     avg = np.average(train_df.loc[train_df['market_segment'] == i].loc[:, 'adr'].to_numpy())
#     num = np.average(len(train_df.loc[train_df['market_segment'] == i].loc[:, 'adr'].to_numpy()))
#     avg_list.append(avg)
#     num_list.append(num)
#     print("average of {} = {}, which has {} samples".format(i, avg, num))

# plt.plot(market_segment, avg_list, 'r.')
# # plt.plot(market_segment, num_list, 'b.')
# plt.show()



# #  --- distribution_channel ---
# distribution_channel = train_df.loc[:, 'distribution_channel']
# plt.plot(distribution_channel, adr, ',')
# plt.ylim(0, 1000)
# plt.show()

# distribution_channel = list(set(distribution_channel))
# distribution_channel.sort()
# avg_list = []
# num_list = []
# for i in distribution_channel:
#     avg = np.average(train_df.loc[train_df['distribution_channel'] == i].loc[:, 'adr'].to_numpy())
#     num = np.average(len(train_df.loc[train_df['distribution_channel'] == i].loc[:, 'adr'].to_numpy()))
#     avg_list.append(avg)
#     num_list.append(num)
#     print("average of {} = {}, which has {} samples".format(i, avg, num))

# plt.plot(distribution_channel, avg_list, 'r.')
# # plt.plot(distribution_channel, num_list, 'b.')
# plt.show()



# #  --- is_repeated_guest ---
# is_repeated_guest = train_df.loc[:, 'is_repeated_guest']
# plt.plot(is_repeated_guest, adr, ',')
# plt.ylim(0, 1000)
# plt.show()

# is_repeated_guest = list(set(is_repeated_guest))
# is_repeated_guest.sort()
# avg_list = []
# num_list = []
# for i in is_repeated_guest:
#     avg = np.average(train_df.loc[train_df['is_repeated_guest'] == i].loc[:, 'adr'].to_numpy())
#     num = np.average(len(train_df.loc[train_df['is_repeated_guest'] == i].loc[:, 'adr'].to_numpy()))
#     avg_list.append(avg)
#     num_list.append(num)
#     print("average of {} = {}, which has {} samples".format(i, avg, num))

# plt.plot(is_repeated_guest, avg_list, 'r.')
# # plt.plot(is_repeated_guest, num_list, 'b.')
# plt.show()



# #  --- previous_cancellations ---
# previous_cancellations = train_df.loc[:, 'previous_cancellations']
# plt.plot(previous_cancellations, adr, ',')
# plt.ylim(0, 1000)
# plt.show()

# previous_cancellations = list(set(previous_cancellations))
# previous_cancellations.sort()
# avg_list = []
# num_list = []
# for i in previous_cancellations:
#     avg = np.average(train_df.loc[train_df['previous_cancellations'] == i].loc[:, 'adr'].to_numpy())
#     num = np.average(len(train_df.loc[train_df['previous_cancellations'] == i].loc[:, 'adr'].to_numpy()))
#     avg_list.append(avg)
#     num_list.append(num)
#     print("average of {} = {}, which has {} samples".format(i, avg, num))

# plt.plot(previous_cancellations, avg_list, 'r.')
# # plt.plot(previous_cancellations, num_list, 'b.')
# plt.show()


# #  --- previous_bookings_not_canceled ---
# previous_bookings_not_canceled = train_df.loc[:, 'previous_bookings_not_canceled']
# plt.plot(previous_bookings_not_canceled, adr, ',')
# plt.ylim(0, 1000)
# plt.show()

# previous_bookings_not_canceled = list(set(previous_bookings_not_canceled))
# previous_bookings_not_canceled.sort()
# avg_list = []
# num_list = []
# for i in previous_bookings_not_canceled:
#     avg = np.average(train_df.loc[train_df['previous_bookings_not_canceled'] == i].loc[:, 'adr'].to_numpy())
#     num = np.average(len(train_df.loc[train_df['previous_bookings_not_canceled'] == i].loc[:, 'adr'].to_numpy()))
#     avg_list.append(avg)
#     num_list.append(num)
#     print("average of {} = {}, which has {} samples".format(i, avg, num))

# plt.plot(previous_bookings_not_canceled, avg_list, 'r.')
# # plt.plot(previous_bookings_not_canceled, num_list, 'b.')
# plt.show()


# #  --- reserved_room_type ---
# reserved_room_type = train_df.loc[:, 'reserved_room_type']
# plt.plot(reserved_room_type, adr, ',')
# plt.ylim(0, 1000)
# plt.show()

# reserved_room_type = list(set(reserved_room_type))
# reserved_room_type.sort()
# avg_list = []
# num_list = []
# for i in reserved_room_type:
#     avg = np.average(train_df.loc[train_df['reserved_room_type'] == i].loc[:, 'adr'].to_numpy())
#     num = np.average(len(train_df.loc[train_df['reserved_room_type'] == i].loc[:, 'adr'].to_numpy()))
#     avg_list.append(avg)
#     num_list.append(num)
#     print("average of {} = {}, which has {} samples".format(i, avg, num))

# plt.plot(reserved_room_type, avg_list, 'r.')
# # plt.plot(reserved_room_type, num_list, 'b.')
# plt.show()



# #  --- assigned_room_type ---
# assigned_room_type = train_df.loc[:, 'assigned_room_type']
# plt.plot(assigned_room_type, adr, ',')
# plt.ylim(0, 1000)
# plt.show()

# assigned_room_type = list(set(assigned_room_type))
# assigned_room_type.sort()
# avg_list = []
# num_list = []
# for i in assigned_room_type:
#     avg = np.average(train_df.loc[train_df['assigned_room_type'] == i].loc[:, 'adr'].to_numpy())
#     num = np.average(len(train_df.loc[train_df['assigned_room_type'] == i].loc[:, 'adr'].to_numpy()))
#     avg_list.append(avg)
#     num_list.append(num)
#     print("average of {} = {}, which has {} samples".format(i, avg, num))

# plt.plot(assigned_room_type, avg_list, 'r.')
# # plt.plot(assigned_room_type, num_list, 'b.')
# plt.show()



# #  --- booking_changes ---
# booking_changes = train_df.loc[:, 'booking_changes']
# plt.plot(booking_changes, adr, ',')
# plt.ylim(0, 1000)
# plt.show()

# booking_changes = list(set(booking_changes))
# booking_changes.sort()
# avg_list = []
# num_list = []
# for i in booking_changes:
#     avg = np.average(train_df.loc[train_df['booking_changes'] == i].loc[:, 'adr'].to_numpy())
#     num = np.average(len(train_df.loc[train_df['booking_changes'] == i].loc[:, 'adr'].to_numpy()))
#     avg_list.append(avg)
#     num_list.append(num)
#     print("average of {} = {}, which has {} samples".format(i, avg, num))

# plt.plot(booking_changes, avg_list, 'r.')
# # plt.plot(booking_changes, num_list, 'b.')
# plt.show()


# #  --- deposit_type ---
# deposit_type = train_df.loc[:, 'deposit_type']
# plt.plot(deposit_type, adr, ',')
# plt.ylim(0, 1000)
# plt.show()

# deposit_type = list(set(deposit_type))
# deposit_type.sort()
# avg_list = []
# num_list = []
# for i in deposit_type:
#     avg = np.average(train_df.loc[train_df['deposit_type'] == i].loc[:, 'adr'].to_numpy())
#     num = np.average(len(train_df.loc[train_df['deposit_type'] == i].loc[:, 'adr'].to_numpy()))
#     avg_list.append(avg)
#     num_list.append(num)
#     print("average of {} = {}, which has {} samples".format(i, avg, num))

# plt.plot(deposit_type, avg_list, 'r.')
# # plt.plot(deposit_type, num_list, 'b.')
# plt.show()



# #  --- agent ---
# # need to remove nan
# new_train_df = train_df.loc[:, ['agent', 'adr']].dropna()
# agent = new_train_df.loc[:, 'agent']
# adr_agent = new_train_df.loc[:, 'adr']
# # agent = train_df.loc[:, 'agent']
# plt.plot(agent, adr_agent, ',')
# plt.ylim(0, 1000)
# plt.show()

# agent = list(set(agent))
# agent.sort()
# avg_list = []
# num_list = []
# for i in agent:
#     avg = np.average(train_df.loc[train_df['agent'] == i].loc[:, 'adr'].to_numpy())
#     num = np.average(len(train_df.loc[train_df['agent'] == i].loc[:, 'adr'].to_numpy()))
#     avg_list.append(avg)
#     num_list.append(num)
#     print("average of {} = {}, which has {} samples".format(i, avg, num))

# plt.plot(agent, avg_list, 'r.')
# # plt.plot(agent, num_list, 'b.')
# plt.show()



# #  --- company ---
# # need to remove nan
# new_train_df = train_df.loc[:, ['company', 'adr']].dropna()
# company = new_train_df.loc[:, 'company']
# adr_company = new_train_df.loc[:, 'adr']
# # company = train_df.loc[:, 'company']
# plt.plot(company, adr_company, ',')
# plt.ylim(0, 1000)
# plt.show()

# company = list(set(company))
# company.sort()
# avg_list = []
# num_list = []
# for i in company:
#     avg = np.average(train_df.loc[train_df['company'] == i].loc[:, 'adr'].to_numpy())
#     num = np.average(len(train_df.loc[train_df['company'] == i].loc[:, 'adr'].to_numpy()))
#     avg_list.append(avg)
#     num_list.append(num)
#     print("average of {} = {}, which has {} samples".format(i, avg, num))

# plt.plot(company, avg_list, 'r.')
# # plt.plot(company, num_list, 'b.')
# plt.show()


# #  --- days_in_waiting_list ---
# days_in_waiting_list = train_df.loc[:, 'days_in_waiting_list']
# plt.plot(days_in_waiting_list, adr, ',')
# plt.ylim(0, 1000)
# plt.show()

# days_in_waiting_list = list(set(days_in_waiting_list))
# days_in_waiting_list.sort()
# avg_list = []
# num_list = []
# for i in days_in_waiting_list:
#     avg = np.average(train_df.loc[train_df['days_in_waiting_list'] == i].loc[:, 'adr'].to_numpy())
#     num = np.average(len(train_df.loc[train_df['days_in_waiting_list'] == i].loc[:, 'adr'].to_numpy()))
#     avg_list.append(avg)
#     num_list.append(num)
#     print("average of {} = {}, which has {} samples".format(i, avg, num))

# plt.plot(days_in_waiting_list, avg_list, 'r.')
# # plt.plot(days_in_waiting_list, num_list, 'b.')
# plt.show()



# #  --- customer_type ---
# customer_type = train_df.loc[:, 'customer_type']
# plt.plot(customer_type, adr, ',')
# plt.ylim(0, 1000)
# plt.show()

# customer_type = list(set(customer_type))
# customer_type.sort()
# avg_list = []
# num_list = []
# for i in customer_type:
#     avg = np.average(train_df.loc[train_df['customer_type'] == i].loc[:, 'adr'].to_numpy())
#     num = np.average(len(train_df.loc[train_df['customer_type'] == i].loc[:, 'adr'].to_numpy()))
#     avg_list.append(avg)
#     num_list.append(num)
#     print("average of {} = {}, which has {} samples".format(i, avg, num))

# plt.plot(customer_type, avg_list, 'r.')
# # plt.plot(customer_type, num_list, 'b.')
# plt.show()



# #  --- required_car_parking_spaces ---
# required_car_parking_spaces = train_df.loc[:, 'required_car_parking_spaces']
# plt.plot(required_car_parking_spaces, adr, ',')
# plt.ylim(0, 1000)
# plt.show()

# required_car_parking_spaces = list(set(required_car_parking_spaces))
# required_car_parking_spaces.sort()
# avg_list = []
# num_list = []
# for i in required_car_parking_spaces:
#     avg = np.average(train_df.loc[train_df['required_car_parking_spaces'] == i].loc[:, 'adr'].to_numpy())
#     num = np.average(len(train_df.loc[train_df['required_car_parking_spaces'] == i].loc[:, 'adr'].to_numpy()))
#     avg_list.append(avg)
#     num_list.append(num)
#     print("average of {} = {}, which has {} samples".format(i, avg, num))

# plt.plot(required_car_parking_spaces, avg_list, 'r.')
# # plt.plot(required_car_parking_spaces, num_list, 'b.')
# plt.show()


# #  --- total_of_special_requests ---
# total_of_special_requests = train_df.loc[:, 'total_of_special_requests']
# plt.plot(total_of_special_requests, adr, ',')
# plt.ylim(0, 1000)
# plt.show()

# total_of_special_requests = list(set(total_of_special_requests))
# total_of_special_requests.sort()
# avg_list = []
# num_list = []
# for i in total_of_special_requests:
#     avg = np.average(train_df.loc[train_df['total_of_special_requests'] == i].loc[:, 'adr'].to_numpy())
#     num = np.average(len(train_df.loc[train_df['total_of_special_requests'] == i].loc[:, 'adr'].to_numpy()))
#     avg_list.append(avg)
#     num_list.append(num)
#     print("average of {} = {}, which has {} samples".format(i, avg, num))

# plt.plot(total_of_special_requests, avg_list, 'r.')
# # plt.plot(total_of_special_requests, num_list, 'b.')
# plt.show()






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

