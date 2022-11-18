import numpy as np
import matplotlib.pyplot as plt
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

#性別がmale/femaleの文字列なため処理が面倒 male=1 female=0に置き換える
sex_converted_df=train_set.select(pl.when(pl.col("Sex")=="male").then(1).otherwise(0).alias("Sex"))
#性別を変換したものと客室の等級の情報を合体する
prepared_set=sex_converted_df.with_column(train_set["Pclass"])

PClassPlot=prepared_set.select(pl.col(["Pclass","Sex"])).groupby("Pclass")
PClassPlot=PClassPlot.mean()
print(PClassPlot["Pclass"])

ax1.bar(x=PClassPlot["Pclass"], height=PClassPlot["Sex"])
ax1.set_ylabel('Proportion of Male')
ax1.set_xlabel('PClass')
ax1.set_yticks(np.arange(0, 1.1,.1))
ax1.set_title("Class and Gender Ratio")

plt.show() 
