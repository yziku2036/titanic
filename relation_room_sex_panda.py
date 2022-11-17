import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing  import LabelEncoder
import polars as pl

# データを読み込む
train_set = pd.read_csv('data/train.csv')
test_set = pd.read_csv('data/test.csv')
#pol=pl.read_csv('data/train.csv')
#print(pol)
fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

print(type(train_set['Age']))
# PClass （旅客等級）
PClassPlot = train_set['Survived'].groupby(train_set['Pclass']).mean()
ax1.bar(x=PClassPlot.index, height=PClassPlot.values)
ax1.set_ylabel('Survival Rate')
ax1.set_xlabel('PClass')
ax1.set_xticks(PClassPlot.index)
ax1.set_yticks(np.arange(0, 1.1,.1))
ax1.set_title("Class and Survival Rate")

# Sex （性別）
GenderPlot = train_set['Survived'].groupby(train_set['Sex']).mean()
ax2.bar(x=GenderPlot.index, height=GenderPlot.values)
ax2.set_ylabel('Survival Rate')
ax2.set_xlabel('Gender')
ax2.set_xticks(GenderPlot.index)
ax2.set_yticks(np.arange(0, 1.1,.1))
ax2.set_title('Gender and Survival Rate')


# Sex （性別）の値を処理(female=0/male=1に変換)
labelencoder=LabelEncoder()
train_set['Sex'] = labelencoder.fit_transform(train_set['Sex'])

ClassPlot = train_set['Sex'].groupby(train_set['Pclass']).size()
print(ClassPlot)
ax2.bar(x=ClassPlot.index, height=ClassPlot.values)
ax2.set_ylabel('PClass')
ax2.set_xlabel('Gender')
ax2.set_xticks(ClassPlot.index)
#ax2.set_yticks(np.arange(0, 1.1,.1))
ax2.set_title('Cabin and Survival Rate')


plt.show() 