import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def main():
# データを読み込む
    train_set = pd.read_csv('data/train.csv')
    test_set = pd.read_csv('data/test.csv')

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(111)
    #ax2 = fig.add_subplot(132)
    #ax3 = fig.add_subplot(133)

    data=train_set.dropna(subset=['Age']).sort_values("Age")

    # 年齢
    #AgePlot = train_set['Survived'].groupby(train_set['Age']).mean()
    AgePlot.index=AgePlot.index.astype(int)
    ax1.bar(x=AgePlot.index, height=AgePlot.values)
    ax1.set_ylabel('Survival Rate')
    ax1.set_xlabel('Age')
    ax1.set_xticks(AgePlot.index)
    ax1.set_yticks(np.arange(0, 1.1,.1))
    ax1.set_title("Class and Survival Rate")

    print(train_set['Survived'])

    plt.show()


if __name__=='__main__':
    main()