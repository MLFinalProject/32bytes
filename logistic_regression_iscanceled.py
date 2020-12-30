import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from dataset import Dataset

hotel = Dataset()

x = hotel.get_dataset(to_numpy = True)
y = hotel.get_is_canceled(to_numpy = True)


reg = LogisticRegression(max_iter=800).fit(x, y)
print(reg.score(x, y))