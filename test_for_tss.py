# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:55:52 2017

#==============================================================================
# TEST FOT TIME_SERIES_SPLIT
#==============================================================================


@author: Dancsek
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import TimeSeriesSplit,cross_val_score, train_test_split 
from sklearn.metrics import accuracy_score
from pandas import datetime

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

df = df.sort_values(by="Formatted Date")

# az indexelés az idő szerint
# helyettesítés szükséges? 
#df.set_index("Formatted Date", inplace=False, );
ts = df.copy()


#%%
ts.set_index("Formatted Date", inplace=True, );

y = ts.Summary.cat.codes
y = y.astype("category")
X = ts
X = X.drop(["Summary"], axis=1)

#y = df.Summary.cat.codes
#y = y.astype("category")
#X = df
#X = X.drop(["Summary"], axis=1)

for tau in ["year","month","day","hour","Precip_Type"]:
    # sajnos a sparse összeszarja magát ha datetime az index típusa
    X = pd.get_dummies(X, prefix=tau ,columns=[tau], sparse = False)
X.columns



#%%
#==============================================================================
# A z  idő indexeléssel akadtak problémák

# nem értem hogy tudom sornként elérni az adatokat. 
# de egyszerre többet már sikerül, ha intervallumot írok be:
ts[datetime(2008,11,21):datetime(2008,11,23)]



# emellett nem világos, hogy a sorrendiségnek van e jelentősége, illetve vajon
# teljesen folytonos a mérési adatok



#==============================================================================


#%%
trainsize = int(len(X.index)*0.67)
train, test = X[0:trainsize], X[trainsize:len(X.index)]
plt.plot(train["Humidity"])
plt.plot(test["Humidity"])
    # a kettőt egyszerre ábrázolni
    # nem működik
    # plt.plot([None for i in train["Humidity"]] + [x for x in test["Humidity"]])
    # plt.show()
    

# nem tudja a dt típust indexelő cuccokkal kezelni
# szükség volna olyan dolgokra, ami felveszik az időintervallumot jellemző információkat,
# úgymint, kezdő és vég pont, mintavételi idő, kihagyások, kb mint egy describe() csak idő dimenzióban.







#%%
plt.plot([train["Humidity"] +  test["Humidity"]])
plt.show()



#%% felbontás évek szerint

x_ = X
for ev in pd.date_range( start=min(X.index), end=max(X.index),freq='H', closed= None):
    print(ev, pd.Timestamp( ev))
    
    X_ = X[pd.Timestamp(ev)]
    print("X_"+str(ev.year))
    x_.rename("X_"+str(ev.year))
    
    
#%%

# HOGY A FASZBA' INDEXELÜNK?????
""" 
x["2005"]
X[2005]
X[PD.DATETIME]
X[PD.TIMERANGE]
X[DATEITEM.DATETIME]
X[pd.Timestamp(ev)] <-- ennek int kell
??
"""
# ez működik:
X[str(ev)].head()
    
#%%
for ev in range(2005,2017,1):
    stamp = pd.Timestamp(str(ev))
    try:
        print(ev, stamp, X[pd.datetime(ev)], X[pd.date_range], X[pd.timedelta_range], X[datetime.date], )
    except:
        print("hiba, találj ki mást")

     
    
    
    
    
    
    
    
    
#%%
# timeseriesplit

Xhead = X.iloc[X.index.indexer_between_time(datetime.time(min(X.index)),datetime.time(max(X.index))),[2]]

tss = TimeSeriesSplit(n_splits=2 )

sindex = tss.split(X)
for tr,tst in sindex:
    print(" train  %s  test  %s " %(tr,tst) )
    
# úgy tűnik a datetime típússal van baja


#%%
X.index.indexer_between_time[datetime.time("2006"),datetime.time("2007")]



