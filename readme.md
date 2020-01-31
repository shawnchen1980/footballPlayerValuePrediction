# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 22:33:39 2019

@author: shawn
"""

1 查看不同的列所包含的空值
df.info()
df.iloc[:,:10].info()
df.isna().sum()
df.info(verbose=True,null_counts=True)
df.describe()
2 去除包含空值的列
df=df.dropna(axis=1)

3 对某一列空值或所有空值进行填充
df[col].fillna(val,inplace=True)
df.fillna(0,inplace=True)
4 对分类型字段进行数值化转换LabelEncoder

5 数值型字段进行正则化处理

6 数据拆分成训练集和测试集

7 数据