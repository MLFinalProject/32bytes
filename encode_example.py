from encoder import *
from feature_engineering import *
from dataset import Dataset

hotel = Dataset()
room_feature = gen_room_feature(hotel.get_feature(['reserved_room_type', 'assigned_room_type']))
net_canceled_feature = gen_net_canceled_feature(hotel.get_feature(['previous_cancellations', 'previous_bookings_not_canceled']))
hotel.add_feature(room_feature)
hotel.add_feature(net_canceled_feature)
hotel.remove_feature(['children', 'stays_in_weekend_nights'])

encode_item = ['country']
encode_ref = ['adr']
input_df = hotel.get_feature(encode_item)
input_df = pd.concat([input_df,hotel.get_train_adr()],axis = 1)
binary_target = 5

#input_df = pd.concat([ input_df[encode_item],backward_difference_encode(input_df[encode_item],input_df[encode_ref]) ],axis = 1)
#input_df = pd.concat([ input_df[encode_item],count_encode(input_df[encode_item],input_df[encode_ref]) ],axis = 1)
#input_df = pd.concat([ input_df[encode_item],helmert_encode(input_df[encode_item],input_df[encode_ref]) ],axis = 1)
#input_df = pd.concat([ input_df[encode_item],leave_one_out_encode(input_df[encode_item],input_df[encode_ref]) ],axis = 1)
#input_df = pd.concat([ input_df[encode_item],m_estimate_encode(input_df[encode_item],input_df[encode_ref],binary_target) ],axis = 1)
#input_df = pd.concat([ input_df[encode_item],one_hot_encode(input_df[encode_item],input_df[encode_ref]) ],axis = 1)
#input_df = pd.concat([ input_df[encode_item],polynomial_encode(input_df[encode_item],input_df[encode_ref]) ],axis = 1)
input_df = pd.concat([ input_df[encode_item],target_encode(input_df[encode_item],input_df[encode_ref]) ],axis = 1)
#input_df = pd.concat([ input_df[encode_item],weight_of_evidence_encode(input_df[encode_item],input_df[encode_ref],binary_target) ],axis = 1)
print( input_df )