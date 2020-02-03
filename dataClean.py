# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:07:03 2019

@author: shawn
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vapeplot import vapeplot
from adjustText import adjust_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score #common metris to evaluate regression models
from sklearn import preprocessing
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.cluster import KMeans

from sklearn.decomposition import IncrementalPCA
# Create linear regression object
from sklearn import linear_model

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from matplotlib.font_manager import FontProperties
#coding:utf-8
#import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
sns.set_style('whitegrid',{'font.sans-serif':'SimHei'})

cols=[
 'short_name',
 'long_name',
 'age',
 'dob',
 'height_cm',
 'weight_kg',
 'nationality',
 'club',
 'overall',
 'potential',
 'value_eur',
 'wage_eur',
 'player_positions',
 'preferred_foot',
 'international_reputation',
 'weak_foot',
 'skill_moves',
 'work_rate',
 'body_type',
 'player_tags',
 'team_position',
 'team_jersey_number',
 'joined',
 'nation_position',
 'nation_jersey_number',
 'pace',
 'shooting',
 'passing',
 'dribbling',
 'defending',
 'physic',
 'gk_diving',
 'gk_handling',
 'gk_kicking',
 'gk_reflexes',
 'gk_speed',
 'gk_positioning',
 'player_traits',
 'attacking_crossing',
 'attacking_finishing',
 'attacking_heading_accuracy',
 'attacking_short_passing',
 'attacking_volleys',
 'skill_dribbling',
 'skill_curve',
 'skill_fk_accuracy',
 'skill_long_passing',
 'skill_ball_control',
 'movement_acceleration',
 'movement_sprint_speed',
 'movement_agility',
 'movement_reactions',
 'movement_balance',
 'power_shot_power',
 'power_jumping',
 'power_stamina',
 'power_strength',
 'power_long_shots',
 'mentality_aggression',
 'mentality_interceptions',
 'mentality_positioning',
 'mentality_vision',
 'mentality_penalties',
 'mentality_composure',
 'defending_marking',
 'defending_standing_tackle',
 'defending_sliding_tackle',
 'goalkeeping_diving',
 'goalkeeping_handling',
 'goalkeeping_kicking',
 'goalkeeping_positioning',
 'goalkeeping_reflexes']

colsA=[
 'short_name',
 'long_name',
 'nationality',
 'club']



colsC=[
 'club',
 'age',
 'height_cm',
 'weight_kg',
 'overall',
 'potential',
 'value_eur',
 'wage_eur',
 'international_reputation',
 'weak_foot',
 'skill_moves',
 'shooting',
 'passing',
 'dribbling',
 'defending',
 'physic',
 'attacking_crossing',
 'attacking_finishing',
 'attacking_heading_accuracy',
 'attacking_short_passing',
 'attacking_volleys',
 'skill_dribbling',
 'skill_curve',
 'skill_fk_accuracy',
 'skill_long_passing',
 'skill_ball_control',
 'movement_acceleration',
 'movement_sprint_speed',
 'movement_agility',
 'movement_reactions',
 'movement_balance',
 'power_shot_power',
 'power_jumping',
 'power_stamina',
 'power_strength',
 'power_long_shots',
 'mentality_aggression',
 'mentality_interceptions',
 'mentality_positioning',
 'mentality_vision',
 'mentality_penalties',
 'mentality_composure',
 'defending_marking',
 'defending_standing_tackle',
 'defending_sliding_tackle',
 'goalkeeping_diving',
 'goalkeeping_handling',
 'goalkeeping_kicking',
 'goalkeeping_positioning',
 'goalkeeping_reflexes']

colsD=[
 'age',
 'height_cm',
 'weight_kg',
 'value_eur',
 'wage_eur',
 'international_reputation',
 'weak_foot',
 'skill_moves',
 'attacking_crossing',
 'attacking_finishing',
 'attacking_heading_accuracy',
 'attacking_short_passing',
 'attacking_volleys',
 'skill_dribbling',
 'skill_curve',
 'skill_fk_accuracy',
 'skill_long_passing',
 'skill_ball_control',
 'movement_acceleration',
 'movement_sprint_speed',
 'movement_agility',
 'movement_reactions',
 'movement_balance',
 'power_shot_power',
 'power_jumping',
 'power_stamina',
 'power_strength',
 'power_long_shots',
 'mentality_aggression',
 'mentality_interceptions',
 'mentality_positioning',
 'mentality_vision',
 'mentality_penalties',
 'mentality_composure',
 'defending_marking',
 'defending_standing_tackle',
 'defending_sliding_tackle',
 'goalkeeping_diving',
 'goalkeeping_handling',
 'goalkeeping_kicking',
 'goalkeeping_positioning',
 'goalkeeping_reflexes']
#如何看df的每一列是什么类型？
#df.dtypes

#如何向df中某个位置插入一列
#df.insert(loc=新列的位置,column=列名,value=整数或者序列)

#如何看df中某一列中是否包含某字符串
#df.col.str.contains("xxx",regex=False)

#如何看df中不同列的数据类型
#df.dtypes
#kmeans分类
#https://github.com/jcalcutt/projects/blob/master/kmeans_clustering/kmeans_fifa.ipynb
#回归
#https://blog.uruit.com/soccer-and-machine-learning-tutorial/
#分析
#https://jiayiwangjw.github.io/2017/06/12/SoccerDataAnalysis/
#https://www.kaggle.com/roshansharma/fifa-data-visualization
#fifa术语解释说明，位置缩写解释
#https://www.fifauteam.com/fifa-ultimate-team-definitions-and-abbreviations/
def toValue(str):
    if(str.find("+")>=0):
        arr=str.split("+")
        return int(arr[0])+int(arr[1])
    elif(str.find("-")>=0):
        arr=str.split("-")
        return int(arr[0])-int(arr[1])
    else:
        return int(str)
    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
import numpy as np
#输出的列中只有两列是非数值型的：club与short_name
def prepareData():
    df1=pd.read_csv('datasets/players_20.csv',index_col=0)
    df1["team_position"].fillna("unknown",inplace=True)
    df1=df1.dropna(axis=1)
    numCol=[]
    for col in df1.columns:
        if(df1[col].dtype!=np.object or (df1[col].dtype==np.object and is_number(df1[col].iloc[0]))):
            df1[col]=df1[col].apply(lambda x : eval(str(x)))
            numCol.append(col)
#    df1_y=df1["value_eur"]
#    df1_x=df1.loc[:,numCol]
#    df1_x=df1_x.drop(columns=["value_eur","wage_eur"])
    return df1.loc[df1['value_eur']>0,['nationality','club','short_name','team_position',*numCol]]

def getClusters(df,clusters):
    df1 = df.groupby('club').transform(lambda x: (x - x.mean()) / x.std())
    df1=df1.drop(columns=["overall","potential","value_eur","wage_eur",'international_reputation'])
    df1=df1.fillna(df1.mean())
    x = df1.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()  # could also test using the StandardScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    X_norm = pd.DataFrame(x_scaled)
    pca = sklearnPCA(n_components=2) #2-dimensional PCA
    transformed = pd.DataFrame(pca.fit_transform(X_norm))
    # Number of clusters
    kmeans = KMeans(n_clusters=clusters)
    # Fitting the input data
    kmeans = kmeans.fit(transformed)
    # Getting the cluster labels
    labels = kmeans.predict(transformed)
    return labels
def prepareDataWithCluster():
    df=prepareData()
    labels=getClusters(df,3)
    df1=df.drop(columns=["overall","potential","value_eur","wage_eur",'international_reputation','nationality','club','short_name','team_position'])
    df1=df1.fillna(df1.mean())
    min_max_scaler = preprocessing.MinMaxScaler()
    df1[df1.columns] = min_max_scaler.fit_transform(df1[df1.columns])
    df1['cluster']=labels
    df1["value_eur"]=df["value_eur"].tolist()
    return df1,df


def plotHist(data):
    from scipy.stats import norm
    #plot the histogram
    plt.hist(data, bins=16, normed=True, alpha=0.6, color='g')
    plt.title("#Players per Overall")
    plt.xlabel("Overall")
    plt.ylabel("Count")
    data_mean = data.mean()
    data_std = data.std()
    # Plot the probability density function for norm
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, data_mean, data_std)
    plt.plot(x, p, 'k', linewidth=2, color='r')
    title = "#Players per Overall, Fit results: mean = %.2f,  std = %.2f" % (data_mean, data_std)
    plt.title(title)
    
    plt.show()
    
def linearReg1(data):
#    data=data.drop(columns=["wage_eur"])
    train, test = train_test_split(data, test_size=0.20, random_state=0)

    xtrain = train[['overall']]
    ytrain = train[['value_eur']]
    xtest = test[['overall']]
    ytest = test[['value_eur']]
#    xtrain=train.drop(columns=["value_eur"])
#    xtest=test.drop(columns=["value_eur"])
    
    regr = linear_model.LinearRegression()
    regr.fit(xtrain, ytrain) 
    # Make predictions using the testing set
    y_pred = regr.predict(xtest)
    plt.scatter(xtest, ytest,  color='black')
    plt.plot(xtest, y_pred, color='blue', linewidth=3)
    plt.xlabel("Overall")
    plt.ylabel("Value")
    plt.show()
#    print(regr.cof_)


    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(ytest, y_pred))
    
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(ytest, y_pred))
    
def linearReg2(data):
    data=data.drop(columns=["wage_eur",'international_reputation','club','short_name','team_position'])
    
    train, test = train_test_split(data, test_size=0.20, random_state=0)

#    xtrain = train[['overall']]
    ytrain = train[['value_eur']]
#    xtest = test[['overall']]
    ytest = test[['value_eur']]
    xtrain=train.drop(columns=["value_eur"])
    xtest=test.drop(columns=["value_eur"])
    
    regr = linear_model.LinearRegression()
    regr.fit(xtrain, ytrain) 
    # Make predictions using the testing set
    y_pred = regr.predict(xtest)
#    plt.scatter(xtest, ytest,  color='black')
#    plt.plot(xtest, y_pred, color='blue', linewidth=3)
#    plt.xlabel("Overall")
#    plt.ylabel("Value")
#    plt.show()
#    print(regr.coef_)
#    for a in zip(xtrain.columns,regr.coef_[0]):
#        print(a)

    print(sorted(zip(xtrain.columns,regr.coef_[0]),key=lambda x:x[1],reverse=True))
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(ytest, y_pred))
    
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(ytest, y_pred))
    
def linearReg3(data,cluster_label=0):
#    data=data.drop(columns=["wage_eur",'international_reputation','club','short_name','team_position'])
    df=data[data["cluster"]==cluster_label].copy()
    df=df.drop(columns=["cluster"])
    
#    for f in df.columns:
#        related = df['value_eur'].corr(df[f])
#        print("%s: %f" % (f,related))
        
    train, test = train_test_split(df, test_size=0.20, random_state=0)

#    xtrain = train[['overall']]
    ytrain = train[['value_eur']]
#    xtest = test[['overall']]
    ytest = test[['value_eur']]
    xtrain=train.drop(columns=["value_eur"])
    xtest=test.drop(columns=["value_eur"])
    
    regr = linear_model.LinearRegression()
    regr.fit(xtrain, ytrain) 
    # Make predictions using the testing set
    y_pred = regr.predict(xtest)
#    plt.scatter(xtest, ytest,  color='black')
#    plt.plot(xtest, y_pred, color='blue', linewidth=3)
#    plt.xlabel("Overall")
#    plt.ylabel("Value")
#    plt.show()
#    print(regr.coef_)
#    for a in zip(xtrain.columns,regr.coef_[0]):
#        print(a)
    dd=sorted(zip(xtrain.columns,regr.coef_[0]),key=lambda x:x[1],reverse=True)
    data=data.loc[:,[dd[0][0],dd[1][0],dd[2][0],dd[3][0],dd[4][0],dd[-1][0],"value_eur"]]
    print(dd)
    fig, ax = plt.subplots()
    ax.scatter(ytest, y_pred)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(ytest, y_pred))
    
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(ytest, y_pred))
    return data,ytest,y_pred

def linearReg4(data,test_index):
    if("cluster" in data.columns):
        data=data.drop(columns=["cluster"])
#    train, test = train_test_split(data, test_size=0.20, random_state=0)
    test=data.loc[test_index,:]
    train=data.loc[np.setdiff1d(data.index,test_index),:]
#    xtrain = train[['overall']]
    ytrain = train[['value_eur']]
#    xtest = test[['overall']]
    ytest = test[['value_eur']]
    xtrain=train.drop(columns=["value_eur"])
    xtest=test.drop(columns=["value_eur"])
    
    regr = linear_model.LinearRegression()
    regr.fit(xtrain, ytrain) 
    # Make predictions using the testing set
    y_pred = regr.predict(xtest)
    print("from linearReg4:Mean squared error: %.2f" % mean_squared_error(ytest, y_pred))
    
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(ytest, y_pred))
    


#from sklearn.preprocessing import PolynomialFeatures
#from sklearn.pipeline import make_pipeline
def polyReg3(data,cluster_label=0):
#    data=data.drop(columns=["wage_eur",'international_reputation','club','short_name','team_position'])
    df=data[data["cluster"]==cluster_label]
    df=df.drop(columns=["cluster"])
    train, test = train_test_split(df, test_size=0.20, random_state=0)

#    xtrain = train[['overall']]
    ytrain = train[['value_eur']]
#    xtest = test[['overall']]
    ytest = test[['value_eur']]
    xtrain=train.drop(columns=["value_eur"])
    xtest=test.drop(columns=["value_eur"])
    
    pol = make_pipeline(PolynomialFeatures(1), linear_model.Ridge())
    pol.fit(xtrain, ytrain)
    # Make predictions using the testing set
    y_pred = pol.predict(xtest)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(ytest, y_pred))
    
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(ytest, y_pred))
    return df

def polyReg4(data):
#    data=data.drop(columns=["wage_eur",'international_reputation','club','short_name','team_position'])
#    df=data[data["cluster"]==cluster_label]
#    df=df.drop(columns=["cluster"])
    if("cluster" in data.columns):
        data=data.drop(columns=["cluster"])
    train, test = train_test_split(data, test_size=0.20, random_state=0)

#    xtrain = train[['overall']]
    ytrain = train[['value_eur']]
#    xtest = test[['overall']]
    ytest = test[['value_eur']]
    xtrain=train.drop(columns=["value_eur"])
    xtest=test.drop(columns=["value_eur"])
    
    pol = make_pipeline(PolynomialFeatures(1), linear_model.Ridge())
    pol.fit(xtrain, ytrain)
    # Make predictions using the testing set
    y_pred = pol.predict(xtest)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(ytest, y_pred))
    
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(ytest, y_pred))
    return df

from sklearn.svm import SVR

def SVR3(data,cluster_label=-1):
#    data=data.drop(columns=["wage_eur",'international_reputation','club','short_name','team_position'])
    df=data
    if(cluster_label>=0):
        df=data[data["cluster"]==cluster_label]
    if("cluster" in df.columns):
        df=df.drop(columns=["cluster"])
    train, test = train_test_split(df, test_size=0.20, random_state=0)

#    xtrain = train[['overall']]
    ytrain = train[['value_eur']]
#    xtest = test[['overall']]
    ytest = test[['value_eur']]
    xtrain=train.drop(columns=["value_eur"])
    xtest=test.drop(columns=["value_eur"])
    
    svr_rbf = SVR(kernel='rbf', gamma=1e-3, C=100, epsilon=0.1)
    svr_rbf.fit(xtrain, ytrain.values.ravel())
#    pol = make_pipeline(PolynomialFeatures(2), linear_model.Ridge())
#    pol.fit(xtrain, ytrain)
    # Make predictions using the testing set
    y_pred = svr_rbf.predict(xtest)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(ytest, y_pred))
    
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(ytest, y_pred))
    return df


def scatterPlot(transformed):
    sns.set(style="white")
    pal =  sns.blend_palette(vapeplot.palette('vaporwave'))
    
    ax = sns.lmplot(x="x", y="y",hue='cluster', data=transformed, legend=False,
                       fit_reg=False, height =8, scatter_kws={"s": 25}, palette=pal)
    
    texts = []
    for x, y, s in zip(transformed.x, transformed.y, transformed.pos):
        texts.append(plt.text(x, y, s))
    adjust_text(texts) #, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))  # uncomment to add arrows to labels
    
    #ax._legend.set_title(prop={fontsize:'15'})
    ax.set(ylim=(-2, 2))
    plt.tick_params(labelsize=15)
    #plt.setp(ax.get_legend().get_title(), fontsize='15')
    plt.xlabel('PC1', fontsize=20)
    plt.ylabel("PC2", fontsize=20)
    plt.show()
    
    
def pcaScatterPlot(df):
    df1 = df.groupby('club').transform(lambda x: (x - x.mean()) / x.std())
    df1=df1.drop(columns=["overall","potential","value_eur","wage_eur",'international_reputation'])
    df1=df1.fillna(df1.mean())
    x = df1.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()  # could also test using the StandardScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    X_norm = pd.DataFrame(x_scaled)
    pca = sklearnPCA(n_components=2) #2-dimensional PCA
    transformed = pd.DataFrame(pca.fit_transform(X_norm))
    # Number of clusters
    kmeans = KMeans(n_clusters=3)
    # Fitting the input data
    kmeans = kmeans.fit(transformed)
    # Getting the cluster labels
    labels = kmeans.predict(transformed)
#    print(df['team_position'])
    transformed['cluster']=labels
    transformed['pos']=df['team_position'].tolist()
    transformed.columns=['x','y','cluster','pos']
    scatterPlot(transformed[:100])
    return transformed,df1

  
def kmeans(df,cluster=3):
    df1 = df.groupby('club').transform(lambda x: (x - x.mean()) / x.std())
    df1=df1.fillna(df1.mean())
    x = df1.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()  # could also test using the StandardScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    X_norm = pd.DataFrame(x_scaled)
    pca = sklearnPCA(n_components=2) #2-dimensional PCA
    transformed = pd.DataFrame(pca.fit_transform(X_norm))
    
    #以下代码为作图代码
#    cluster_range = range( 1, 11 )
#    cluster_errors = []
#    for num_clusters in cluster_range:
#      clusters = KMeans( num_clusters )
#      clusters.fit( transformed )
#      cluster_errors.append( clusters.inertia_ )
#    clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )
#    sns.set(style="white")
#    plt.figure(figsize=(12,6))
#    plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )
#    plt.tick_params(labelsize=15)
#    
#    plt.xlabel("Clusters", fontsize=20)
#    plt.ylabel("Sum of squared errors", fontsize=20)
    #以上代码为作图代码
    
    # Number of clusters
    kmeans = KMeans(n_clusters=cluster)
    # Fitting the input data
    kmeans = kmeans.fit(transformed)
    # Getting the cluster labels
    labels = kmeans.predict(transformed)
    # Centroid values
    C = kmeans.cluster_centers_
    clusters = kmeans.labels_.tolist()
    df["cluster"]=clusters
    return df

def polyRidge(df):
    df1=df.drop(columns=["wage_eur",'international_reputation','club','short_name','team_position'])
    train, test = train_test_split(df1, test_size=0.20, random_state=0)
    xtrain = train[['overall']]
    ytrain = train[['value_eur']]
    xtest = test[['overall']]
    ytest = test[['value_eur']]
#    xtrain=train.drop(columns=["value_eur"])
#    xtest=test.drop(columns=["value_eur"])
    pol = make_pipeline(PolynomialFeatures(3), linear_model.Ridge())
    pol.fit(xtrain, ytrain)
    y_pol = pol.predict(xtest)
    plt.scatter(xtest, ytest,  color='black')
    plt.scatter(xtest, y_pol,  color='blue')
    plt.xlabel("Value")
    plt.ylabel("Overall")
    plt.show()
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(ytest, y_pol))
    
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(ytest, y_pol))



#有中文出现的情况，需要u'内容'

#df1=df.groupby(by='team_position').size().reset_index(name='counts')

#针对数据df绘制不同位置球员数量的柱状图
def posCountBarplot(df):
    fig,ax=plt.subplots()
    df['team_position'].value_counts().plot(kind='bar',ax=ax)
    
#左中场与后卫速度比较
def compareStVsCB(df):
    fig,ax=plt.subplots(1,2,sharex=True,sharey=True)
    df1=df[(df['team_position']=='LM')|(df['team_position']=='CB')]
    df1.boxplot(by='team_position',ax=ax,column=['movement_sprint_speed','defending_sliding_tackle'])
    ax[0].set_title(u'冲刺速度')
    ax[0].xaxis.set_label_text(u'场上位置')
    ax[0].set_xticklabels([u'中后卫',u'左中场'])
    ax[1].set_title(u'防守铲断')
    ax[1].xaxis.set_label_text(u'场上位置')
    fig.suptitle(u'场上不同位置球员参数比较',y=1.02)    
#关于subplots作图的参考文献
#http://jonathansoma.com/lede/algorithms-2017/classes/fuzziness-matplotlib/how-pandas-uses-matplotlib-plus-figures-axes-and-subplots/
#https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/subplots_demo.html
#巴西与中国球员参数比较
def compareChinaVsBrazil(df):
    fig,ax=plt.subplots(1,2,sharex=True,sharey=True)  # if use subplot
    df1=df[((~df['team_position'].isin(['RCB','LCB','LB','RB','CB']))&(df['nationality']=='China PR'))|((df['team_position'].isin(['RCB','LCB','LB','RB','CB']))&(df['nationality']=='Brazil'))]
    df1.boxplot(by='nationality',ax=ax,column=['movement_sprint_speed','defending_sliding_tackle'])
    ax[0].set_title(u'冲刺速度')
    ax[0].xaxis.set_label_text(u'国籍与位置')
    ax[0].set_xticklabels([u'巴西后卫',u'中国中前场'])
    ax[1].set_title(u'防守铲断')
    ax[1].xaxis.set_label_text(u'国籍与位置')
    fig.suptitle(u'巴西与中国球员参数比较',y=1.02)
#    fig.subplots_adjust(vspace=0.5,y=1.08)
#    fig.title(u'标题')

def compareClusteredVsUnclustered(df):
    data0,t0,p0=linearReg3(df,cluster_label=0)
    data1,t1,p1=linearReg3(df,cluster_label=1)
    data2,t2,p2=linearReg3(df,cluster_label=2)
    test=pd.concat([t0,t1,t2])
    pred=np.concatenate((p0,p1,p2))
    print("先分类后预测获得的结果Mean squared error: %.2f" % mean_squared_error(test, pred))
        
        # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(test, pred))
    linearReg4(df,test.index)   

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

def larger_model():
	# create model
	model = Sequential()
	model.add(Dense(40, input_dim=40, kernel_initializer='normal', activation='relu'))
	model.add(Dense(40, kernel_initializer='normal', activation='relu'))
	model.add(Dense(40, kernel_initializer='normal', activation='relu'))
	model.add(Dense(40, kernel_initializer='normal', activation='relu'))
#	model.add(Dense(40, kernel_initializer='normal', activation='relu'))
#	model.add(Dense(40, kernel_initializer='normal', activation='relu'))
#	model.add(Dense(40, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

def deepLearning(df):
    X = df.iloc[:,0:-1]   
    y = df.iloc[:,-1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)
    estimator = KerasRegressor(build_fn=larger_model, epochs=1200, batch_size=100, verbose=True)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    #2 hidden layers one40nodes, one20nodes
    #epochs=2400
    #Mean squared error: 7119022716725.74
    #Variance score: 0.76
    #epochs=3600
    #Mean squared error: 5576111544862.35
    #Variance score: 0.81
    #epochs=1200
    #Mean squared error: 12008907570558.37
    #Variance score: 0.59
    
#epochs=4800
#Mean squared error: 4423903632738.11
#Variance score: 0.85
    
#4 hidden layers,each 40nodes	
#epoch=1200
#Mean squared error: 1721517292241.19
#Variance score: 0.94
    
#epoch=1200
#7 hidden layers, each 40 nodes
#Mean squared error: 2152989680037.98
#Variance score: 0.93
    print("Mean squared error: %.2f"
          % mean_squared_error(y_test, y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_test, y_pred))  

df,df2=prepareDataWithCluster()

#作图比较中场球员与后卫的技术指标
#compareStVsCB(df2)

#作图比较中国中前场与巴西后卫技术指标
#compareChinaVsBrazil(df2)

#作图画出主成分分析中不同位置球员在二维坐标坐标中的位置（比较费时间）
#pcaScatterPlot(df2)

#分类与不分类分别使用线性回归对球员身价进行预测,可以看出分类后mse和r2值都明显上升
#compareClusteredVsUnclustered(df)

#不断更改epoch值进行深度学习
deepLearning(df)

#[('movement_reactions', 10669910.629105326), ('defending_standing_tackle', 8233411.759554121), ('defending_sliding_tackle', 6384230.850897608), ('mentality_interceptions', 4542779.73979069), ('defending_marking', 4184467.7363923676), ('mentality_vision', 2780563.2569291545), ('movement_sprint_speed', 2482640.1234917007), ('attacking_finishing', 2113008.841974644), ('mentality_composure', 1629612.8627936952), ('weight_kg', 1277576.0351028857), ('power_jumping', 1249848.2550068246), ('attacking_short_passing', 1198090.2160034678), ('mentality_aggression', 1096285.1837503752), ('mentality_positioning', 914432.8799217364), ('skill_curve', 748474.2926481327), ('attacking_heading_accuracy', 722945.6335170204), ('height_cm', 651538.2113331575), ('attacking_volleys', 641160.4266911634), ('skill_long_passing', 464998.1387351558), ('weak_foot', 448929.49055599596), ('goalkeeping_reflexes', 373743.67539316276), ('movement_acceleration', 317273.2748238363), ('skill_moves', 178901.477029702), ('skill_dribbling', 172342.93207938666), ('skill_fk_accuracy', 64721.92788577685), ('movement_balance', -221523.83494034846), ('goalkeeping_diving', -244265.33573962498), ('power_shot_power', -256187.3344564906), ('goalkeeping_kicking', -275932.0002267789), ('power_strength', -456039.4460197133), ('power_stamina', -651478.7889918883), ('skill_ball_control', -1077709.9220470565), ('mentality_penalties', -1374244.9787419515), ('power_long_shots', -1487943.7426576086), ('movement_agility', -1640604.8456918402), ('goalkeeping_positioning', -1668536.0140711204), ('goalkeeping_handling', -2294265.0389139345), ('attacking_crossing', -2315619.190697673), ('age', -6925056.75244279)]
#Mean squared error: 10355848813632.86
#Variance score: 0.52
#[('skill_ball_control', 11724765.379985083), ('movement_reactions', 10061238.007560737), ('mentality_positioning', 9100275.84520762), ('attacking_finishing', 6646647.135305739), ('movement_balance', 5088508.454718586), ('mentality_vision', 4722836.310373804), ('mentality_composure', 4149489.271548888), ('attacking_volleys', 3685000.6567562083), ('attacking_short_passing', 3589523.4026997536), ('height_cm', 3441249.4942992646), ('mentality_penalties', 2956969.5739450566), ('movement_sprint_speed', 2562100.2371546924), ('movement_acceleration', 2147270.3131488813), ('skill_long_passing', 1272779.1799433194), ('weight_kg', 1268839.8278496824), ('goalkeeping_diving', 1236364.7048546178), ('power_stamina', 720340.5715701429), ('weak_foot', 533045.9790597907), ('skill_fk_accuracy', 501698.4671984357), ('skill_dribbling', 445299.3289277621), ('defending_standing_tackle', 346572.6963168192), ('defending_marking', 291809.4278673781), ('mentality_interceptions', 250151.2041448322), ('attacking_crossing', -17598.488613030873), ('attacking_heading_accuracy', -76307.93643800495), ('goalkeeping_positioning', -116449.15735696061), ('power_long_shots', -159446.4545291015), ('skill_moves', -307022.45748063247), ('goalkeeping_kicking', -419473.5797119781), ('power_jumping', -483765.54781315196), ('power_strength', -615158.1943065776), ('skill_curve', -618177.1861645863), ('goalkeeping_reflexes', -655476.1730211015), ('defending_sliding_tackle', -755489.9710008588), ('mentality_aggression', -771612.9470113758), ('power_shot_power', -1334486.4313025065), ('goalkeeping_handling', -3006895.878114899), ('movement_agility', -3559250.0372709734), ('age', -9916975.348692529)]
#Mean squared error: 20965567696215.24
#Variance score: 0.51
#[('power_shot_power', 17172460.776008468), ('goalkeeping_handling', 16465508.550456032), ('goalkeeping_diving', 6743546.26133606), ('goalkeeping_reflexes', 6510927.7569581), ('goalkeeping_positioning', 4727551.350920109), ('defending_marking', 3425983.205654784), ('attacking_short_passing', 3394099.7616823814), ('attacking_crossing', 2975443.287281673), ('movement_reactions', 2833771.2118568914), ('power_stamina', 1946135.342450929), ('mentality_aggression', 1932920.423006175), ('skill_long_passing', 1916322.403712281), ('movement_sprint_speed', 1231255.8972011951), ('movement_balance', 1052228.9967413363), ('mentality_composure', 865917.731450093), ('attacking_volleys', 803269.3994347525), ('skill_ball_control', 793029.1735847051), ('mentality_vision', 788809.7952409254), ('weight_kg', 662075.8442018786), ('attacking_heading_accuracy', 416240.2909501824), ('movement_agility', 358499.46260358626), ('power_jumping', 306271.3380701011), ('skill_dribbling', 246708.2081508837), ('skill_moves', 3.4063123166561127e-07), ('movement_acceleration', -429129.8125893632), ('defending_standing_tackle', -502230.2136016511), ('height_cm', -512745.1149534881), ('mentality_interceptions', -539173.7623797043), ('weak_foot', -620736.6893642079), ('mentality_positioning', -877346.393627662), ('power_strength', -1173745.2097248957), ('power_long_shots', -1265520.4605342778), ('skill_fk_accuracy', -1458744.2337713372), ('skill_curve', -2006835.458327919), ('defending_sliding_tackle', -2549148.5177886635), ('mentality_penalties', -2741166.6190425614), ('age', -7260340.123803136), ('attacking_finishing', -8800218.784083473), ('goalkeeping_kicking', -11465047.700804586)]
#Mean squared error: 19773334204704.18
#Variance score: 0.44
#先分类后预测获得的结果Mean squared error: 16277087533829.73
#Variance score: 0.51
#from linearReg4:Mean squared error: 18339070013284.68
#Variance score: 0.45

#print("poly reg start")
##
#polyReg3(df,cluster_label=0)
#polyReg3(df,cluster_label=1)
#polyReg3(df,cluster_label=2)
#
#polyReg4(df)
#
#
##SVR3(data0,cluster_label=-1)
##SVR3(data1,cluster_label=-1)
##SVR3(data2,cluster_label=-1)
##
##SVR3(df,cluster_label=-1)
##dd=pcaScatterPlot(df)
#
#
##df1,df2=pcaScatterPlot(df)
##df1.plot(x=0,y=1,kind='scatter')
##for i in range(3):
##    print('printing cluster {}',i)
##    posCountBarplot(df[df['cluster']==i])
#    
##plotHist(df["overall"])
##linearReg2(df)
#
##linearReg1(df)
#
#colsB=[
# 'nationality',
# 'club']
#df1=pd.read_csv('datasets/players_20.csv',index_col=0)
#mkdict = lambda row: dict([(col, row[col]) for col in colsB])
#df1=df1[colsB].apply(mkdict, axis=1)
#df=kmeans(df)
#df=linearReg3(df,cluster_label=1)
#linearReg4(df)

#polyRidge(df)
#df12 = df12.groupby('club').transform(lambda x: (x - x.mean()) / x.std())
#df1=df11.join(df12)
        
