# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:07:03 2019

@author: shawn
"""

import pandas as pd

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

colsC=[
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
 'gk_diving',
 'gk_handling',
 'gk_kicking',
 'gk_reflexes',
 'gk_speed',
 'gk_positioning',
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

def toValue(str):
    if(str.find("+")>=0):
        arr=str.split("+")
        return int(arr[0])+int(arr[1])
    elif(str.find("-")>=0):
        arr=str.split("-")
        return int(arr[0])-int(arr[1])
    else:
        return int(str)
    
import numpy as np
df1=pd.read_csv('datasets/players_19.csv',index_col=0)
df1=df1.loc[:,colsC]
for col in df1.columns:
    if(df1[col].dtype==np.object):
        df1[col]=df1[col].apply(toValue)
        
df2=pd.read_csv('datasets/players_18.csv',index_col=0)
df2=df2.loc[:,colsC]
for col in df2.columns:
    if(df2[col].dtype==np.object):
        df2[col]=df2[col].apply(toValue)
df3=df1-df2
df3=df3[np.isnan(df3['overall'])==False]