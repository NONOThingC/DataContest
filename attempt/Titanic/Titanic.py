##此文件只是说明哪些步骤以及总结，并不是实际代码，实际代码比较容易找就不再附在其中。

#首先要确定关键特征，可以画相关图
# 使用facetgrid来画相关图
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()

#分析完之后就要进行特征提取，这个步骤包括数值的分析和字符变量的分析，编码转换等等

#之后就可以导入机器学习模型之中了，然后做cross-validation
#步骤基本同我在MCM模板中的代码。