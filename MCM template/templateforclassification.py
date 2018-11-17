# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 17:16:41 2018

@author: hu
"""
# Import necessary modules
from __future__ import print_function
from __future__ import division
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import numpy as np

#filename='.\distribute_2016.08.10_110100_.csv'
#filename='response_2016.08.10_110100_.csv'

"""
#If the file is line based
with open('...') as f:
d    for line in f:
        process(line) # 

file = open(filename)
while 1:
lines = file.readlines(100000)
if not lines:
    break
for line in lines:
    pass # do something
"""
#读XLS
"""
data_xls = pd.read_excel(filename)
data_xls.to_csv('didi.csv', encoding='utf-8')
data=data_xls
"""
#读CSV
#data=pd.read_csv(filename,encoding='utf-8',sep='	',header=None)
data=pd.read_csv(filename,encoding='utf-8')
#大数据读法
"""
data_df=data.get_chunk(100000)
#data_df=pd.concat([data_element for data_element in data if len],ignore_index=True)
print (data.get_chunk(10))
"""
#读txt
#data=pd.read_table(filename,iterator=True,header=None,sep=',',chunksize=10000)




# 生成画布、3D图形对象、三维散点图
"""
import pandas as pd
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure() 
ax = Axes3D(fig) 
ax.scatter(data['longitude'],data['latitude'],data['value'])
#ax.plot(data['longitude'],data['latitude'],zdir='y')
#ax.plot(data['longitude'],data['latitude'],0)
# 设置坐标轴显示以及旋转角度
ax.set_xlabel('lng')
ax.set_ylabel('lat')
ax.set_zlabel('value')
ax.view_init(elev=10,azim=235)
plt.show()
"""

#数据处理




#K 均值算法
#Import Library

#Assumed you have, X (attributes) for training data set and x_test(attributes) of test_dataset
# Create KNeighbors classifier object model
# Train the model using the training sets and check score
#Predict Output



#Kmeans
#Elbow Method决定聚类数目K

"""
from sklearn.cluster import KMeans
distorsions = []
for k in range(2, 20):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    distorsions.append(kmeans.inertia_)

fig = plt.figure(figsize=(15, 5))
plt.plot(range(2, 20), distorsions)
plt.grid(True)
plt.title('Elbow curve')
"""



#Silhouette Method聚类决定K的方法
"""
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
print(__doc__)

# This particular setting has one distinct cluster and 3 clusters placed close
# together.

range_n_clusters = [7,8,9,10,11]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X.iloc[:, 0], X.iloc[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    plt.show()
"""

"""
from sklearn.cluster import KMeans
model=KMeans(n_clusters=8, random_state=9)
y_pred = model.fit_predict(X)
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_pred)
plt.show()
"""



"""
digits = datasets.load_digits()

# Print the keys and DESCR of the dataset
print(digits.DESCR)
print(digits.keys())
# Import necessary modules


# Create feature and target arrays
X = digits.data
y = digits.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state=42, stratify=y)

# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=7)

# Fit the classifier to the training data
knn.fit(X_train,y_train)

# Print the accuracy
print(knn.score(X_test,y_test))
# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit the classifier to the training data
    knn.fit(X_train,y_train)
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)
    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
# Predict the labels for the training data X
y_pred = knn.predict(X)
print ("PredictionGroup: {}".format(y_pred))
"""
#PCA
"""
from sklearn.decomposition import PCA
# Set up PCA and the X vector for diminsionality reduction
pca = PCA()
wine_X = wine.drop("Type", axis=1)
# Apply PCA to the wine dataset X vector
transformed_X = pca.fit_transform(wine_X)
# Look at the percentage of variance explained by the different components
print(pca.explained_variance_ratio_)
"""
#线性回归
"""
#Import other necessary libraries like pandas, numpy...
from sklearn import linear_model

#Load Train and Test datasets
#Identify feature and response variable(s) and values must be numeric and numpy arrays
x_train=input_variables_values_training_datasets
y_train=target_variables_values_training_datasets
x_test=input_variables_values_test_datasets

# Create linear regression object
linear = linear_model.LinearRegression()
# Train the model using the training sets and check score
linear.fit(x_train, y_train)
linear.score(x_train, y_train)
#Equation coefficient and Intercept
print('Coefficient: n', linear.coef_)
print('Intercept: n', linear.intercept_)
#Predict Output
predicted= linear.predict(x_test)
"""
#逻辑回归
"""
from sklearn.linear_model import LogisticRegression
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create logistic regression object
model = LogisticRegression()
# Train the model using the training sets and check score
model.fit(X, y)
model.score(X, y)
#Equation coefficient and Intercept
print('Coefficient: n', model.coef_)
print('Intercept: n', model.intercept_)
#Predict Output
predicted= model.predict(x_test)
"""
#决策树
"""
#Import Library
#Import other necessary libraries like pandas, numpy...
from sklearn import tree

#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create tree object
model = tree.DecisionTreeClassifier(criterion='gini') # for classification, here you can change the algorithm as gini or entropy (information gain) by default it is gini
# model = tree.DecisionTreeRegressor() for regression
# Train the model using the training sets and check score
model.fit(X, y)
model.score(X, y)
#Predict Output
predicted= model.predict(x_test)
"""
#支持向量机
"""
#Import Library
from sklearn import svm
#Assumed you have, X (predic
tor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object
model = svm.svc() # there is various option associated with it, this is simple for classification. You can refer link, for mo# re detail.
# Train the model using the training sets and check score
model.fit(X, y)
model.score(X, y)
#Predict Output
predicted= model.predict(x_test)
"""
#朴素贝叶斯
"""
#Import Library
from sklearn.naive_bayes import GaussianNB
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object model = GaussianNB() # there is other distribution for multinomial classes like Bernoulli Naive Bayes, Refer link
# Train the model using the training sets and check score
model.fit(X, y)
#Predict Output
predicted= model.predict(x_test)
"""
#KNN
"""
#Import Library
from sklearn.neighbors import KNeighborsClassifier
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create KNeighbors classifier object model
KNeighborsClassifier(n_neighbors=6) # default value for n_neighbors is 5
# Train the model using the training sets and check score
model.fit(X, y)
#Predict Output
predicted= model.predict(x_test)
"""

#K 均值算法
"""
#Import Library
from sklearn.cluster import KMeans

#Assumed you have, X (attributes) for training data set and x_test(attributes) of test_dataset
# Create KNeighbors classifier object model
k_means = KMeans(n_clusters=3, random_state=0)

# Train the model using the training sets and check score
model.fit(X)

#Predict Output
predicted= model.predict(x_test)
"""
#随机森林
"""
#Import Library
from sklearn.ensemble import RandomForestClassifier

#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create Random Forest object
model= RandomForestClassifier()

# Train the model using the training sets and check score
model.fit(X, y)

#Predict Output
predicted= model.predict(x_test)
"""
#降维算法
"""
#Import Library
from sklearn import decomposition

#Assumed you have training and test data set as train and test
# Create PCA obeject pca= decomposition.PCA(n_components=k) #default value of k =min(n_sample, n_features)
# For Factor analysis
#fa= decomposition.FactorAnalysis()
# Reduced the dimension of training dataset using PCA
train_reduced = pca.fit_transform(train)

#Reduced the dimension of test dataset
test_reduced = pca.transform(test)

#For more detail on this, please refer  this link.
"""
#Gradient Boosting 和 AdaBoost 算法
"""
#Import Library
from sklearn.ensemble import GradientBoostingClassifier
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create Gradient Boosting Classifier object
model= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
# Train the model using the training sets and check score
model.fit(X, y)
#Predict Output
predicted= model.predict(x_test)
"""
#线性回归
"""
# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.3, random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test , y_pred))
print("Root Mean Squared Error: {}".format(rmse))

"""
"""
# Import the necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


# Create a linear regression object: reg
reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg,X , y,cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

"""
#Logistic回归
"""

# Import the necessary modules
    from sklearn.metrics import roc_curve
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix , classification_report
    # Create training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

    # Create the classifier: logreg
    logreg = LogisticRegression()

    # Fit the classifier to the training data
    logreg.fit(X_train, y_train)

    # Predict the labels of the test set: y_pred
    y_pred = logreg.predict(X_test)

    # Compute and print the confusion matrix and classification report
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
"""

#岭回归


#效果验证

# 以下为分类变量
"""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix , classification_report
# Predict the labels of the test data: y_pred
y_pred = knn.score(X_test,  y_test)
# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# Import necessary modules
from sklearn.metrics import roc_curve 

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test , y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
"""

#写入文件
"""
savename="informationDay"+str()+".csv"
information.to_csv(savename,index=True,sep=',')
"""


#校赛第二次示例代码
"""
tiaojian=(data['longitude']<116.1 )|( data['longitude']>116.9 )|( data['latitude']>40.4)|(data['latitude']<39.5)
tiaojian=~tiaojian
data=data[tiaojian]

yy=[]
for i in range(24):
    yy.append(data[data['hour']==i].shape[0])
plt.plot(range(24),yy)
plt.xlabel('time/h')
plt.ylabel('demand')
#plt.ylabel('response')
#data.plot(x='longitude',y='latitude',kind='scatter')
yihuannei=(data['longitude']>116.363)&( data['longitude']<116.441 )&( data['latitude']>39.906)&(data['latitude']<39.955 )
sanhuannei=(data['longitude']>116.316 )&( data['longitude']<116.468)&( data['latitude']>39.863 )&(data['latitude']< 39.974)
wuhuannei=(data['longitude']>116.217)&( data['longitude']<116.550)&( data['latitude']> 39.762)&(data['latitude']< 40.029)
wuhuanwai=~wuhuannei
data['area']=-1
data.loc[wuhuanwai,'area']=1 #wu
data.loc[wuhuannei,'area']=4
data.loc[sanhuannei,'area']=3
data.loc[yihuannei,'area']=2
plt.figure(2)
plt.scatter(data['longitude'], data['latitude'], c=data['area'])
plt.xlabel('longitude')
plt.ylabel('latitude')
#plt.ylabel('response')
plt.show()
X=pd.concat([data['longitude'],data['latitude']],axis=1)
#Center_Point=np.array([,])
"""