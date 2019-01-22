# learning SVM
这个文档使用了latex支持。
<br>**定理 1 训练集 T 线性可分的充要条件是，T 的两类样本集$M+$正类集和$M-$负类集的凸包相离。**
>凸包（Convex Hull）是一个计算几何（图形学）中的概念。
在一个实数向量空间V中，对于给定集合X，所有包含X的凸集的交集S被称为X的凸包。X的凸包可以用X内所有点(X1，...Xn)的凸组合来构造.

>在二维欧几里得空间中，凸包可想象为一条刚好包著所有点的橡皮圈。
用不严谨的话来讲，给定二维平面上的点集，凸包就是将最外层的点连接起来构成的凸多边形，它能包含点集中所有的点。

<br>**定理 2 当训练集样本为线性可分时，存在唯一的规范超平面$\omega ·x+b=0$，使得**
$$
\begin{cases}
\omega \cdot x_i+b\geq 1 & y_i=1 \\
\omega \cdot x_i+b=\leq-1 & y_i=-1
\end{cases}
$$
规范超平面的定义是：
$$
\begin{cases}
y_i\cdot(\omega \cdot x_i+b)\geq 0 & y_i=1 (保证每一侧的值都正确划分)\\
\min\limits_{i=1,...,l}|\omega \cdot x_i+b|=1 & l为所有样本点个数 &(每一侧最近点和这个超平面距离总是1，这里1是规范超平面假设的单位1) \\
\end{cases}
$$
则普通支持向量间的间隔为$\frac{2}{||w||}$,最优超平面即意味着最大化$\frac{2}{||w||}$。
于是原问题就可以转化为求$\min\frac{||w||}{2}=\min\frac{w^2}{2}$
如下图：

![图片1](.\SVM-LinearBorder.png)

这个优化问题可以用其对偶式解决，对偶式思想的提出nifoumi。

这里必须提到KKT条件.这里有一篇对KKT条件理解的[文章](https://www.zhihu.com/question/23311674/answer/235256926)
<br>给定优化问题：
$$
\min\limits_xf(x) \\
s.t. h_j(x)=0,j=1,...,q \\
g_i(x)\leq0,i=1,...,p \\
那么由于空间中取极值点一定在梯度相同点处取得，那么现在如果有一点x^*是满足约束的极值点，可以得到我们需要的极值点处的梯度是在后面函数的梯度的线性组合处：\\
\nabla f(x^*)=\sum_j \lambda_j \nabla h_j(x^*)+\sum_i \alpha_i \nabla g_i(x^*) (1)  \\
\lambda_j可以取实数,因为线性边界并没有规定梯度方向 \\
\alpha_i一定大于等于0,这是因为要和f(x)梯度方向相反，故这里一定要取正(注意(1)的写法)。\\
具体来说可以在g_i(x^*)\leq 0时取大于等于0的值（因为规定了梯度的方向），在g_i(x^*)\geq 0时必须为0.因为此时g_i(x^*)\geq 0对求最小值毫无贡献，反而增加负担，所以\alpha_i取0。\\
$$

为了在一定程度上容忍分错类的情况，这里引入了一次的惩罚项来容忍一定程度的分类错误，优化函数变为：
$$
\min \frac{1}{2}||\omega^2||+C\sum\limits^l_{i=1}\epsilon_i \\
s.t.y_i(\omega\cdot x+b)\geq1-\epsilon  \\
 \epsilon_i\geq0
$$