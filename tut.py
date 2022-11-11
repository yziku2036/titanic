import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# データを読み込む
train_set = pd.read_csv('data/train.csv')
test_set = pd.read_csv('data/test.csv')

print(train_set.head())
train_set.plot()
plt.show()