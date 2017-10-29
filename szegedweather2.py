# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 16:00:21 2017

#==============================================================================
# TEST FOR PREPROCESSING: coding categoricals
#==============================================================================

@author: Dancsek
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

np.random.seed(1)


# nyitás
df = pd.read_csv("U:/adatbanyaszatialkalmazasok/bázisok/szeged-weather/weatherHistory.csv",sep= ',')
features = df.columns


# formázás és  idő változók
df["Formatted Date"] = pd.to_datetime(df["Formatted Date"])
df["month"] = df["Formatted Date"].dt.month
df["day"] = df["Formatted Date"].dt.day
df["year"] = df["Formatted Date"].dt.year
df["hour"] = df["Formatted Date"].dt.hour


# df["hour","day","month","year"].astype("category")

# az indexelés az idő szerint
df.set_index("Formatted Date",inplace = True);

df.Summary.value_counts();
df["Daily Summary"].value_counts();

# df["Summary"].corr(df["Daily Summary"])
df.Summary.value_counts().sum();
df.Summary = df.Summary.astype("category");
# mindenesetre eldobom a Daily Summery-t
Daily_summary = pd.Series(df["Daily Summary"]);
df = df.drop(["Daily Summary"], axis=1);
df = df.drop(["Loud Cover"], axis=1) 


# szerkesztés
df["Precip_Type"] = df["Precip Type"].astype("category");
df = df.drop(["Precip Type"], axis=1)



#%%
        
#split

y = df.Summary.cat.codes
y = y.astype("category")
X = df.drop(["Summary"], axis=1)
X.Precip_Type = X.Precip_Type.cat.codes

#%% modelling & scoring
gbc = GradientBoostingClassifier(verbose=4)
gbc.fit(X,y)
pred = gbc.predict(X)
score_acc = accuracy_score(y, pred)
sns.barplot(X.columns,gbc.feature_importances_)


#%% coding the X train set

Xcoded = X
for tau in ["year","month","day","hour","Precip_Type"]:
    Xcoded = pd.get_dummies(Xcoded, prefix=tau ,columns=[tau])
Xcoded.columns

#%% kísérletezéd sparse mátrixokkal

Xsparse = X
for tau in ["year","month","day","hour","Precip_Type"]:
    Xsparse = pd.get_dummies(Xsparse, prefix=tau ,columns=[tau], sparse= True)
Xsparse.columns



#%% modelling & scoring for the coded train set
gbccoded = GradientBoostingClassifier(verbose=2)
gbccoded.fit(Xcoded,y)
pred_coded = gbccoded.predict(Xcoded)
score_acc_coded = accuracy_score(y, pred_coded)
sns.barplot(Xcoded.columns,gbccoded.feature_importances_)

#%%
if score_acc > score_acc_coded:
    print(" jobb predikciót az "+ "X" +" adta" )
else:
    print (" jobb predikciót a"+ "Xcoded"+"adta" )
    
#%%
gbcsparse = GradientBoostingClassifier(verbose=2)
gbcsparse.fit(Xsparse,y)
pred_sparse = gbcsparse.predict(Xsparse)
score_acc_sparse = accuracy_score(y, pred_sparse)
sns.barplot(Xsparse.columns,gbcsparse.feature_importances_)



#%%
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()  
dtcoded = DecisionTreeClassifier()  
dtsparse = DecisionTreeClassifier()  
"""
elképzelés:
map( fit( [X, Xcoded, Xparse], y), [dt, dtcoded, dtsparse])

 
trees = [dt,dtcoded,dtsparse]
sets = [X,Xcoded,Xsparse]

for tree,x in {trees:sets}:
    tree.fit(x,y)
"""    
dt.fit(X,y)
dtcoded.fit(Xcoded,y)
dtsparse.fit(Xsparse,y)

pred_dt = dt.predict(X)
pred_coded_dt = dtcoded.predict(Xcoded)
pred_sparse_dt = dtsparse.predict(Xsparse)

score_acc_dt = accuracy_score(y, pred_dt,normalize=False)
score_acc_coded_dt = accuracy_score(y, pred_coded_dt,normalize=False)
score_acc_sparse_dt = accuracy_score(y, pred_sparse_dt,normalize=False)

print("dt: "+ str(score_acc_dt), "dt_coded: "+str(score_acc_coded_dt),"dtsparse: "+str(score_acc_sparse_dt),end="\n")

# Elvileg a dt mindent ( mindent !) helyesen osztályozott
# nyílván validálni is kellene... ezt most nem teszem meg nem ez volt a cél
# ...ja persze... azon az y-on mértem vissza amin tanítottam... (˙-˙)



#%%

if y.values == pd.Series(pred_dt, index=y.index).values :
    print("igen")
else: print("nem")