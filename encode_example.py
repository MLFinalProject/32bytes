from encoder import *

input_df = pd.read_csv('./data/train.csv')
encode_item = ['country','meal']
encode_ref = ['adr']
binary_target = 5

#input_df = pd.concat([ input_df[encode_item],backward_difference_encode(input_df[encode_item],input_df[encode_ref]) ],axis = 1)
#input_df = pd.concat([ input_df[encode_item],count_encode(input_df[encode_item],input_df[encode_ref]) ],axis = 1)
#input_df = pd.concat([ input_df[encode_item],helmert_encode(input_df[encode_item],input_df[encode_ref]) ],axis = 1)
#input_df = pd.concat([ input_df[encode_item],leave_one_out_encode(input_df[encode_item],input_df[encode_ref]) ],axis = 1)
#input_df = pd.concat([ input_df[encode_item],m_estimate_encode(input_df[encode_item],input_df[encode_ref],binary_target) ],axis = 1)
#input_df = pd.concat([ input_df[encode_item],one_hot_encode(input_df[encode_item],input_df[encode_ref]) ],axis = 1)
input_df = pd.concat([ input_df[encode_item],polynomial_encode(input_df[encode_item],input_df[encode_ref]) ],axis = 1)
#input_df = pd.concat([ input_df[encode_item],target_encode(input_df[encode_item],input_df[encode_ref]) ],axis = 1)
#input_df = pd.concat([ input_df[encode_item],weight_of_evidence_encode(input_df[encode_item],input_df[encode_ref],binary_target) ],axis = 1)
print( input_df )