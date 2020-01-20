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
    return df1.loc[:,['nationality','club','short_name','team_position',*numCol]]

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
    df=data[data["cluster"]==cluster_label]
    df=df.drop(columns=["cluster"])
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
    data=data.loc[:,[dd[0][0],dd[1][0],dd[2][0],dd[3][0],dd[4][0],"value_eur"]]
    print(dd)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(ytest, y_pred))
    
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(ytest, y_pred))
    return data

def linearReg4(data):
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
    print("Mean squared error: %.2f" % mean_squared_error(ytest, y_pred))
    
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(ytest, y_pred))
    
    
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
    pol = make_pipeline(PolynomialFeatures(6), linear_model.Ridge())
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
df,df2=prepareDataWithCluster()
linearReg3(df,cluster_label=0)
linearReg3(df,cluster_label=1)
linearReg3(df,cluster_label=2)
#dd=pcaScatterPlot(df)


#df1,df2=pcaScatterPlot(df)
#df1.plot(x=0,y=1,kind='scatter')
#for i in range(3):
#    print('printing cluster {}',i)
#    posCountBarplot(df[df['cluster']==i])
    
#plotHist(df["overall"])
#linearReg2(df)

#linearReg1(df)

#df=kmeans(df)
#df=linearReg3(df,cluster_label=1)
#linearReg4(df)

#polyRidge(df)
#df12 = df12.groupby('club').transform(lambda x: (x - x.mean()) / x.std())
#df1=df11.join(df12)
        
