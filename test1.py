import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# データを読み込む
train_set = pd.read_csv('data/train.csv')
test_set = pd.read_csv('data/test.csv')

fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

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

#年齢別
AgePlot = train_set['Survived'].groupby(train_set['Age']).median()
ax3.bar(x=AgePlot.index, height=AgePlot.values)
ax3.set_ylabel('Survival Rate')
ax3.set_xlabel('Age')
ax3.set_xticks(AgePlot.index)
ax3.set_yticks(np.arange(0, 1.1,.1))
ax3.set_title('Age and Survival Rate')
print(AgePlot.index)
print(AgePlot.values)


plt.show()