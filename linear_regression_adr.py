import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from dataset import Dataset

hotel = Dataset()

x = hotel.get_dataset(to_numpy = True)
y = hotel.get_adr(to_numpy = True)

reg = LinearRegression().fit(x, y)
print(reg.score(x, y))