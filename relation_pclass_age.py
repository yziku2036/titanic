import numpy as np
import matplotlib.pyplot as plt
import polars as pl

# データを読み込む
train_set = pl.read_csv('data/train.csv')
test_set = pl.read_csv('data/test.csv')

fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

#sex_converted_df=train_set.select(pl.when(pl.col("Sex")=="male").then(1).otherwise(0).alias("Sex"))
#prepared_set=sex_converted_df.with_column(train_set["Pclass"])
prepared_set=train_set.fill_null(train_set["Age"].median())
PClassPlot=prepared_set.select(pl.col(["Pclass","Age"])).groupby("Pclass")
PClassPlot=PClassPlot.mean()
print(PClassPlot)


ax1.bar(x=PClassPlot["Pclass"], height=PClassPlot["Age"])
ax1.set_ylabel('Age_mean')
ax1.set_xlabel('PClass')
#ax1.set_yticks(np.arange(0, 1.1,.1))
ax1.set_title("Class and Gender Ratio")

plt.show() 
