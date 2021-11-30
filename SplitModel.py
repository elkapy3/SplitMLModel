from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


data = pd.read_csv('data/data.csv')

data.head()


train_data, test_data = train_test_split(data, test_size=0.2)

train_data.head()


train_data, test_data = train_test_split(data, test_size=0.2)

data['price_category'] = pd.cut(data, test_size=0.2)

data.describe()

data['price_category'] = pd.cut(data['AveragePrice'], bins=[
                                0, 0.7, 1.2, 1.6, 2.5, 3., np.inf], labels=[1, 2, 3, 4, 5, 6])


train_data, test_data = train_test_split(data, test_size=0.2)

train_data['price_category'].value_counts()/len(train_data)

test_data['price_category'].value_counts()/len(test_data)


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

for train_ids, test_ids in split.split(data, data['price_category']):
    train_data = data.loc[train_ids]
    test_data = data.loc[test_ids]

test_data['price_category'].value_counts()/len(test_data)
