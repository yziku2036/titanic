import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing  import LabelEncoder
import polars as pl

#https://qiita.com/nkay/items/9cfb2776156dc7e054c8 polas tutrial

# データを読み込む
train_set = pl.read_csv('data/train.csv')
test_set = pl.read_csv('data/test.csv')
#pol=pl.read_csv('data/train.csv')
#print(pol)
fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

prepared_set=train_set.fill_null(train_set["Age"].mean())
print(type(train_set['Age']))
PClassPlot=prepared_set.select(pl.col(["Survived","Pclass","Age"])).groupby("Pclass")


print(PClassPlot.mean())
PClassPlot=PClassPlot.mean()


ax1.bar(x=PClassPlot["Pclass"], height=PClassPlot["Survived"])
ax1.set_ylabel('Survival Rate')
ax1.set_xlabel('PClass')
#ax1.set_xticks(PClassPlot.index)
ax1.set_yticks(np.arange(0, 1.1,.1))
ax1.set_title("Class and Survival Rate")

plt.show() 
