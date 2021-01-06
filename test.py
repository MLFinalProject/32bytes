import pandas as pd

from dataset import Dataset
from classification import *
from regression import *
from feature_engineering import *
from predict import predict


hotel_is_cancel = Dataset()
absolute_peak_transform(hotel_is_cancel.get_feature(["arrival_date_week_number"]), 34)