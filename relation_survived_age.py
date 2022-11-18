import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# データを読み込む
train_set = pd.read_csv('data/train.csv')
test_set = pd.read_csv('data/test.csv')

fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(111)
#ax2 = fig.add_subplot(132)
#ax3 = fig.add_subplot(133)


train_set['Age'].fillna(train_set['Age'].median(), inplace=True)
AgePlot = train_set['Survived'].groupby(train_set['Age']).mean()
print(AgePlot)
ax1.bar(x=AgePlot.index, height=AgePlot.values)
ax1.set_ylabel('Survival Rate')
ax1.set_xlabel('Age')
#ax3.set_xticks(Age_int.index,minor='False')
ax1.set_yticks(np.arange(0, 1.1,.1))    
ax1.set_title('Age and Survival Rate')
#Age_int.plot()
plt.show() 