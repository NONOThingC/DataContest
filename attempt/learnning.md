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