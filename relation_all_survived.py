import numpy as np
import matplotlib.pyplot as plt
import polars as pl



def main():

# データを読み込む
    train_set = pl.read_csv('data/train.csv')
    test_set = pl.read_csv('data/test.csv')
    
    prepared_set=preproc(train_set)
    #fig0=plt.figure(figsize=(8,4))
    fig1 = plt.figure(figsize=(12, 12),tight_layout=True)
    #ax0 = fig0.add_subplot(111)
    ax1 = fig1.add_subplot(331)
    ax2 = fig1.add_subplot(332)
    ax3 = fig1.add_subplot(333)
    ax4 = fig1.add_subplot(334)
    ax5 = fig1.add_subplot(335)
    ax6 = fig1.add_subplot(336)
    ax7 = fig1.add_subplot(337)
    #all_survived(prepared_set,ax1,"PassengerId")
    all_survived(prepared_set,ax1,"Pclass")
    sex_survived(prepared_set,ax2)
    all_survived(prepared_set,ax3,"Age")
    all_survived(prepared_set,ax4,"SibSp")
    all_survived(prepared_set,ax5,"Parch")
    all_survived(prepared_set,ax6,"Fare")
    embarked_survived(prepared_set,ax7)

    plt.show() 

def preproc(data):
    age_fixed=data.fill_null(data['Age'].median())
    prepared_set=age_fixed.with_column(age_fixed["Embarked"].fill_null("S"))
    
    return prepared_set


def embarked_survived(data,ax):
    #性別がmale/femaleの文字列なため処理が面倒 male=1 female=0に置き換える
    
    emb_converted_df=data.with_column(pl.col("Embarked").str.replace("S","0").str.replace("Q","1").str.replace("C","2"))
    prepared_set=emb_converted_df.with_column(pl.col("Embarked").cast(pl.Int64))

    plot=prepared_set.select(pl.col(["Embarked","Survived"])).groupby("Embarked").mean().sort("Embarked")
    print(plot)
    ax.bar(x=plot["Embarked"],height=plot["Survived"])
    ax.set_ylabel('Survived')
    ax.set_xlabel('Embarked')
    ax.set_xticks([0,1,2],['Southampton','Queenstown','Cherbourg'])
    ax.set_yticks(np.arange(0, 1.1,.1))
    ax.set_title('Embarked Place and Survival Rate')

def sex_survived(data,ax):

    #性別がmale/femaleの文字列なため処理が面倒 male=1 female=0に置き換える
    sex_converted_df=data.select(pl.when(pl.col("Sex")=="male").then(1).otherwise(0).alias("Sex"))
    #性別を変換したものと客室の等級の情報を合体する
    prepared_set=sex_converted_df.with_column(data["Survived"])
    plot=prepared_set.select(pl.col(["Sex","Survived"])).groupby("Sex").mean().sort("Sex")
    ax.bar(x=[0,1],height=plot["Survived"])
    ax.set_ylabel('Survived')
    ax.set_xlabel('Gender')
    ax.set_xticks([0,1],['female','male'])
    ax.set_yticks(np.arange(0, 1.1,.1))
    ax.set_title('Gender and Survival Rate')


def all_survived(data,ax,tag):
    plot=data.select(pl.col(["Survived",tag])).groupby(tag).mean().sort(tag)
    plot[tag].cast(pl.Int64)
    print(tag+"'s info:")
    print(plot[tag])    
    ax.bar(x=plot[tag], height=plot["Survived"])
    ax.set_ylabel('Survival Rate')
    ax.set_xlabel(tag)
    ax.set_yticks(np.arange(0, 1.1,.1))
    ax.set_title(tag+" and Survival") 


if __name__=='__main__':
    main()