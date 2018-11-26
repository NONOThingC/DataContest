# 缺失值处理
使用sklearn.preprocessing.Imputer类来处理使用np.nan对缺失值进行编码过的数据集。

代码如下：
```
In [1]: import numpy as np
   ...: from sklearn.preprocessing import Imputer
   ...: train_X = np.array([[1, 2], [np.nan, 3], [7, 6]])
   ...: imp = Imputer(missing_values=np.nan , strategy='mean', axis=0)
   ...: imp.fit(train_X)
   ...:
Out[1]: Imputer(axis=0, copy=True, missing_values=nan, strategy='mean', verbose=
0)
In [2]: imp.statistics_
Out[2]: array([ 4.        ,  3.66666667])
In [3]: test_X = np.array([[np.nan, 2], [6, np.nan], [7, 6]])
   ...: imp.transform(test_X)
   ...:
Out[3]:
array([[ 4.        ,  2.        ],
       [ 6.        ,  3.66666667],
       [ 7.        ,  6.        ]])
In [4]: imp.fit_transform(test_X)
Out[4]:
array([[ 6.5,  2. ],
       [ 6. ,  4. ],
       [ 7. ,  6. ]])
In [5]: imp.statistics_
Out[5]: array([ 6.5,  4. ])
```
Jupyter可以用一个数据的平均值这样的aggregate function去填补其它函数的缺失值。
#XGBOOST
过程直接看这个图即可。
![xgboost image](https://i.imgur.com/e7MIgXk.png)
但是要明白其中一些参数的含义：
* n_estimators规定了进行多少次循环，一般100-1000，根据学习速率不同而不同。
* early_stopping_rounds是规定在validation scores弱化多少次以后就停止循环，一般=5比较好。
* learning_rate不多BB，其实就是梯度下降的步长值（类似，这样说其实不准确）
a small learning rate (and large number of estimators) will yield more accurate XGBoost models, though it will also take the model longer to train since it does more iterations through the cycle
* n_jobs并行任务数，好吧，并不清楚内部原理。后面再说。

# Pipeline
sklearn中的黑箱子并行，
my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())
格式大致如此

# Data Leakage（非常重要的问题）
leakage causes a model to look accurate until you start making decisions with the model, and then the model becomes very inaccurate.
* Leaky Predictors
  你的数据在预测时不再可用。一般发生是在你要预测的项发生在自变量的项之前，也就是说你要预测的项已经测量完了，然后才得到自变量，这样会造成因果混乱，而且这个自变量对要预测量毫无作用，因此这样的量应该删去。
  ![tu](https://i.imgur.com/CN4INKb.png)
* Leaky Validation Strategy
  一般发生在你把测试集也预处理和fit了，把测试集也拿去做交叠划分验证了，这样测试准确性就会很高，但是模型的鲁棒性却很差。

# 重要概念
1. RMSE 的一个很好的特性是，它可以在与原目标相同的规模下解读。一般可以比较 RMSE 与目标最大值和最小值的差值的大小。
2. steps：训练迭代的总次数。一步计算一批样本产生的损失，然后使用该值修改一次模型的权重。
batch size：单步的样本数量（随机选择）。例如，SGD 的批次大小为 1。
以下公式成立：
total number of trained examples=steps*batch isze
3. 准确率Accuary:所有东西中分类对了的概率。
   精确率Precision：认为是对的中真正对的概率。
   recall：真的是正类的情况中分类（认为）对的概率。


# 组合独热矢量（其实就是笛卡儿积）
例子：

假设我们的模型需要根据以下两个特征来预测狗主人对狗狗的满意程度：
* 行为类型（吠叫、叫、偎依等）
* 时段

如果我们根据这两个特征构建以下特征组合：
>  [behavior type X time of day]

我们最终获得的预测能力将远远超过任一特征单独的预测能力。

例如，如果狗狗在下午 5 点主人下班回来时（快乐地）叫喊，可能表示对主人满意度的正面预测结果。如果狗狗在凌晨 3 点主人熟睡时（也许痛苦地）哀叫，可能表示对主人满意度的强烈负面预测结果。