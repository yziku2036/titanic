import numpy as np
import matplotlib.pyplot as plt
import polars as pl



def main():

# データを読み込む
    train_set = pl.read_csv('data/train.csv')
    test_set = pl.read_csv('data/test.csv')
    fig0=plt.figure(figsize=(8,4))
    fig1 = plt.figure(figsize=(12, 4))
    ax0 = fig0.add_subplot(111)
    ax1 = fig1.add_subplot(121)
    ax2 = fig1.add_subplot(122)
    #ax3 = fig.add_subplot(133)

    survived_age(train_set,ax0)
    pclass_age(train_set,ax1)
    pclass_survived(train_set,ax2)

    plt.show() 


def survived_age(data,ax):
    prepared_set=data.fill_null(data['Age'].median())
    AgePlot = prepared_set.select(pl.col(["Survived","Age"])).groupby("Age").mean().sort("Age")
    print(AgePlot)
    ax.bar(x=AgePlot["Age"], height=AgePlot["Survived"])
    ax.set_ylabel('Survival Rate')
    ax.set_xlabel('Age')
    ax.set_yticks(np.arange(0, 1.1,.1))    
    ax.set_title('Age and Survival Rate')
    


def pclass_age(data,ax):
    prepared_set=data.fill_null(data["Age"].median())

    PClassPlot=prepared_set.select(pl.col(["Pclass","Age"])).groupby("Pclass").mean().sort("Pclass")

    print(PClassPlot)
    
    ax.bar(x=PClassPlot["Pclass"], height=PClassPlot["Age"])
    ax.set_ylabel('Age_mean')
    ax.set_xlabel('PClass')
    ax.set_title("Class and Gender Ratio")
    #return PClassPlot

def pclass_survived(data,ax):
    
    prepared_set=data.fill_null(data["Age"].mean())
    print(type(data['Age']))
    PClassPlot=prepared_set.select(pl.col(["Survived","Pclass","Age"])).groupby("Pclass").mean().sort("Pclass")

    print(PClassPlot)

    ax.bar(x=PClassPlot["Pclass"], height=PClassPlot["Survived"])
    ax.set_ylabel('Survival Rate')
    ax.set_xlabel('PClass')
    #ax.set_xticks(PClassPlot.index)
    ax.set_yticks(np.arange(0, 1.1,.1))
    ax.set_title("Class and Survival Rate")



if __name__=='__main__':
    main()