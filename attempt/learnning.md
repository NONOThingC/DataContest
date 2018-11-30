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
batch size：单步的样本数量（随机选择）。
以下公式成立：
total number of trained examples=steps*batch isze
1. 准确率Accuary:所有东西中分类对了的概率。
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

# L1 和 L2 正则化。
不再多BB，其实两种在feature selection中都有应用。L2其实也可以通过给权重设定阈值的方式来进行筛选。

L2 和 L1 采用不同的方式降低权重：

L2 会降低权重^2。
L1 会降低 |权重|。
因此，L2 和 L1 具有不同的导数：

L2 的导数为 2 * 权重。
L1 的导数为 k（一个常数，其值与权重无关）。
您可以将 L2 的导数的作用理解为每次移除权重的 x%。对于任意数字，即使按每次减去 x% 的幅度执行数十亿次减法计算，最后得出的值也绝不会正好为 0。

可以将 L1 的导数的作用理解为每次从权重中减去一个常数。不过，由于减去的是绝对值，L1 在 0 处具有不连续性，这会导致与 0 相交的减法结果变为 0。例如，如果减法使权重从 +0.1 变为 -0.2，L1 便会将权重设为 0。就这样，L1 使权重变为 0 了。

L1 正则化 - 减少所有权重的绝对值 - 证明对宽度模型非常有效。

请注意，该说明适用于一维模型。

# 神经网络
## [神经网络与拟合关系的理解](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)(从拓扑学的角度)
1. 神经网络每一层其实是对每个维度的旋转、扭曲变换。Each layer stretches and squishes space, but it never cuts, breaks, or folds it。Intuitively, we can see that it preserves topological properties.Transformations like this, which don’t affect topology, are called homeomorphisms.
<br>定理如下：如果权重矩阵W是非奇异的，那么有N个输入N个输出的层是同胚的 Layers with N inputs and N outputs are homeomorphisms, if the weight matrix, W, is non-singular.
<br>定义域上连续，反函数定义域上连续，是双射的函数是同胚的。
理解了这些就可以思考为什么以下的A,B不能只用二层或者以下层数的神经网络进行分类。
![classfication](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/img/topology_base.png)
因为你在这两维之中无论怎么拉伸他都没有一条直线穿过。，如下图。
![answer](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/img/topology_2D-2D_train.gif)

2. If a neural network using layers with only 3 units can classify it, then it is an unlink(a bunch of things that are tangled together, but can be separated by continuous deformation).
unlink就是说可以被有限步破坏而导致不连接。
<br>In topology, we would call it an ambient isotopy between the original link and the separated ones.
原始目标和分开的目标之间的关系叫做ambient isotopy环境同位素。
<br>Theorem: There is an ambient isotopy between the input and a network layer’s representation if: a) W isn’t singular, b) we are willing to permute the neurons in the hidden layer, and c) there is more than 1 hidden unit.
这个定理告诉了我们什么样的输入是这样的网络可以解的。
<br>Links and knots are 1-dimensional manifolds, but we need 4 dimensions to be able to untangle all of them. Similarly, one can need yet higher dimensional space to be able to unknot n-dimensional manifolds. All n-dimensional manifolds can be untangled in 2n+2 dimensions
所以n维的unlink至少需要2n+2维才能解开。

3. 文章作者提议其实可以尝试在合理的地方用KNN，因为KNN其实意义比较明确，因此这样的方式可以更好的理解网络。但是，Sadly, even with sophisticated architecture, using k-NN only gets down to 5-4% test error – and using simpler architectures gets worse results. 

4. 神经网络做的自然事情，非常简单的路线，是试图将两个类分开，并尽可能地拉伸缠绕的部分。虽然这不会接近真正的解决方案，但它可以实现相对较高的分类准确度并且是诱人的局部最小值。
<br>它会在它试图拉伸的区域和近乎不连续的区域中表现为非常高的偏差。我们知道这些事情会发生，于是用正则化，惩罚项来应对这个问题。
