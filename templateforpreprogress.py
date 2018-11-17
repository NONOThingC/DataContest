#coding=utf8
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns

# 读取数据
"""
#使用zip和dict来创建dataframe
# Zip the 2 lists together into one list of (key,value) tuples: zipped
zipped = list(zip(list_keys,list_values))

# Inspect the list using print()
print(zipped)

# Build a dictionary with the zipped list: data
data = dict(zipped)

# Build and inspect a DataFrame from the dictionary: df
df = pd.DataFrame(data)
print(df)
"""
filename=
data=pd.read_csv(filename)
#data=pd.read_csv(filename,index_col="Loan_ID")
#查看数据信息
data.info()
data.head()
data.shape
print(df.value_counts(dropna=False)) #参数含义是不计算缺失值

#处理字符串
"""
# Split on the comma to create a list: column_labels_list
column_labels_list = column_labels.split(',')
# Assign the new column labels to the DataFrame: df.columns
df.columns = column_labels_list
# Remove the appropriate columns: df_dropped
df_dropped = df.drop(list_to_drop,axis='columns')
"""
#处理数字
"""
# Create a list of the columns to average
run_columns = ['run'+str(x) for x in range(1,6)]
# Use apply to create a mean column
running_times_5k["mean"] = running_times_5k.apply(lambda x: x[run_columns].mean(), axis=1)
# Take a look at the results
print(running_times_5k)
"""
#处理时间变量
"""
# Convert the date column to string: df_dropped['date']
df_dropped['date'] = df_dropped['date'].astype(str)
# Pad leading zeros to the Time column: df_dropped['Time']
df_dropped['Time'] = df_dropped['Time'].apply(lambda x:'{:0>4}'.format(x))
# Concatenate the new date and Time columns: date_string
date_string = df_dropped['date']+df_dropped['Time']
# Convert the date_string Series to datetime: date_times
date_times = pd.to_datetime(date_string, format='%Y%m%d%H%M')
# Set the index to be the new date_times container: df_clean
df_clean = df_dropped.set_index(date_times)
#清洗行数据 errors='coerce'表示遇到异常数据值就替换成NAN
df_clean['dew_point_faren'] = pd.to_numeric(df_clean['dew_point_faren'], errors='coerce')
"""
#处理日期
"""
# First, convert string column to date column
volunteer["start_date_converted"] = pd.to_datetime(volunteer["start_date_date"])
# Extract just the month from the converted column
volunteer["start_date_month"] = volunteer["start_date_converted"].apply(lambda row: row.month)
# Take a look at the original and new columns
print(volunteer[['start_date_converted' , 'start_date_month']].head())
"""
#处理文本
"""
In [6]: from sklearn.feature_extraction.text import TfidfVectorizer
In [7]: print(documents.head())
0 Building on successful events last summer and ...
1 Build a website for an Afghan business
2 Please join us and the students from Mott Hall...
3 The Oxfam Action Corps is a group of dedicated...
4 Stop 'N' Swap reduces NYC's waste by finding n...
In [8]: tfidf_vec = TfidfVectorizer()
In [9]: text_tfidf = tfidf_vec.fit_transform(documents)
# Write a pattern to extract numbers and decimals
def return_mileage(length):
    pattern = re.compile(r"\d+\.\d+")
    # Search the text for matches
    mile = re.match(pattern, length)
    # If a value is returned, use group(0) to return the found value
    if mile is not None:
        return float(mile.group(0))
# Apply the function to the Length column and take a look at both columns
hiking["Length_num"] = hiking["Length"].apply(lambda row: return_mileage(row))
print(hiking[["Length", "Length_num"]].head())
# Take the title text
title_text = volunteer["title"]
# Create the vectorizer method
tfidf_vec = TfidfVectorizer()
# Transform the text into tf-idf vectors
text_tfidf = tfidf_vec.fit_transform(title_text)
"""
#简单查看初步处理后数据
df_clean.describe()
#数据清洗
"""
#画出PDF,CDF
fig, axes = plt.subplots(nrows=2, ncols=1)
# Plot the PDF
df.fraction.plot(ax=axes[0], kind='hist', normed=True, bins=30, range=(0,.3))
plt.show()
# Plot the CDF
df.fraction.plot(ax=axes[1], normed=True, cumulative=True, bins=30, kind='hist', range=(0,.3))
plt.show()
#画出频率分布图
df.plot(kind='hist', rot=70, logx=True, logy=True) #rot表示坐标轴刻度旋转
df.boxplot(column='initial_cost', by='Borough', rot=90)#by是根据后面参数来分组画图，类似hue
"""
#重采样
#daily_climate = df_climate.resample('D').mean()

# 多变量
sns.pairplot(tips, hue='sex')
plt.show()


#写入CSV
dataframe.to_csv("test.csv",index=False,sep=',')