# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:33:08 2019

@author: shawn
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv('step0.csv')
columns=['order','Team1', 'Opponent','Home', 'Win','Draw','Lose', 'Goal',
       'Shoot', 'ShootT', 'Foul', 'Corner','Yellow','Red']

def step1():
    ndf=pd.DataFrame(columns=columns)
    for i in range(df.shape[0]):
        item1={'order':i*2,'Team1':df['HomeTeam'][i],'Opponent':df['AwayTeam'][i],'Home':1,'Win':int(df['FTR'][i]=='H'),
               'Draw':int(df['FTR'][i]=='D'),'Lose':int(df['FTR'][i]=='A'),'Goal':df['FTHG'][i],'Shoot':df['HS'][i],
               'ShootT':df['HST'][i],'Foul':df['HF'][i],'Corner':df['HC'][i],'Yellow':df['HY'][i],'Red':df['HR'][i]}
        
        item2={'order':i*2+1,'Team1':df['AwayTeam'][i],'Opponent':df['HomeTeam'][i],'Home':0,'Win':int(df['FTR'][i]=='A'),
               'Draw':int(df['FTR'][i]=='D'),'Lose':int(df['FTR'][i]=='H'),'Goal':df['FTAG'][i],'Shoot':df['AS'][i],
               'ShootT':df['AST'][i],'Foul':df['AF'][i],'Corner':df['AC'][i],'Yellow':df['AY'][i],'Red':df['AR'][i]}
        ndf=ndf.append(item1,ignore_index=True)
        ndf=ndf.append(item2,ignore_index=True)
    ndf=ndf.sort_values(by=['Team1','order'])
    ndf['order']=range(ndf.shape[0])
    return ndf
    
def step2(backStep=3):
    ndf=step1()
    teams=list(set(df['HomeTeam']))
    teams.sort()
   
    cols=['Win','Draw','Lose', 'Goal',
       'Shoot', 'ShootT', 'Foul', 'Corner','Yellow','Red']
    res=pd.DataFrame(columns=cols)
    for team in teams:
        df1=ndf[ndf['Team1']==team].sort_values(by=['order'])
        #print(type(df1.loc[:,cols].apply(lambda x:x.shift().rolling(backStep).sum())))
        res=res.append(df1.loc[:,cols].apply(lambda x:x.shift().rolling(backStep).sum()))
    res.columns=[col+'X' for col in cols]
    res=pd.concat([ndf,res],axis=1)
    res1=res[res['Home']==1]
#    print(res1)
    
    res2=res[res['Home']==0]
#    print(res2)
    res=pd.merge(res1,res2,left_on=['Team1','Opponent'],right_on=['Opponent','Team1'])
    return res
#        win=win+(arr)
#        arr=list(df1['Draw'].shift().rolling(backStep).sum())
#        draw=draw+(arr)
#        arr=list(df1['Win'].shift().rolling(backStep).sum())
#        win=win+(arr)
#    print(win)
#    ndf['WinX']=win
#    return ndf
def step3(backStep=3):
    ndf=step2(backStep)
    cols=['Team1_x','Opponent_x','Win_x','Draw_x','Lose_x']+[col for col in ndf.columns if ('X_' in col) and ('Draw' not in col) ]
    ndf=ndf[pd.notnull(ndf['WinX_x'])].loc[:,cols]
    return ndf

def step4():
    his=pd.read_csv('history.csv')
    hisRank=pd.read_csv('14-15Rank.csv',index_col=0)
    his['HisWin']=[(1 if his.iloc[i,9]=='H' else 0) for i in range(his.shape[0])]
    his['HisLose']=[(1 if his.iloc[i,9]=='A' else 0) for i in range(his.shape[0])]
    his['HomeRank']=[hisRank.loc[his.iloc[i,2],'Points'] for i in range(his.shape[0])]
    his['AwayRank']=[hisRank.loc[his.iloc[i,3],'Points'] for i in range(his.shape[0])]
    return his.loc[:,['HomeTeam','AwayTeam','HisWin','HisLose','HomeRank','AwayRank']]
        
def step5():
    his=step4()
    df=step3()
    res=pd.merge(df,his,left_on=['Team1_x','Opponent_x'],right_on=['HomeTeam','AwayTeam'])
    return res

df1=step5()

df1.to_csv('result.csv',index=False)
dataset=df1
#X=dataset.iloc[:,-4:].values
X=dataset.iloc[:,5:-6].values
y=dataset.iloc[:,2].values
y=y.astype('int')

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#from sklearn.linear_model import LogisticRegression
#classifier=LogisticRegression(random_state=0)

#from sklearn.tree import DecisionTreeClassifier
#classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)

#from sklearn.naive_bayes import GaussianNB
#classifier=GaussianNB()

#from sklearn.ensemble import RandomForestClassifier
#classifier=RandomForestClassifier(n_estimators=20,criterion='entropy',random_state=0)

#from sklearn.neighbors import KNeighborsClassifier
#classifier=KNeighborsClassifier(n_neighbors=5, metric='minkowski',p=2)

from sklearn.svm import SVC
classifier=SVC(kernel='linear',random_state=0)

import keras
from keras.models import Sequential
from keras.layers import Dense



classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)