# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 12:03:54 2017

@author: User
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor


import pdp_per_window


df = pd.read_excel("U:/adatbanyaszatialkalmazasok/bázisok/Példa-pdp-v1.xlsx" ,sheetname= 1)
df_deletable = df[["Deletable_1","Szorzo1","Szorzo2"]]
df = df.drop(["Deletable_1","Szorzo1","Szorzo2"], axis = 1   )

df = df.sort_values(by = "Time")
df["Time"] = pd.DatetimeIndex(df["Time"].values, freq= pd.infer_freq(df["Time"]))
df.set_index("Time",inplace = True,)

lir = LinearRegression()
lor = LogisticRegression()
gbr = GradientBoostingRegressor()

#%%
y = df.Target
X = df.drop(["Target"], axis=1)

tstart = X.index.min()
tend = X.index[10]
T = X.index.max()
toffset = 7

#%%


try:
    pd_figs , predfram = pdp_per_window.roll_predict(tstart, tend, X, toffset, T, lor, y)
    pd_figs
except: 
    print("\n szar a lekvárban")
#%%

 pd_figs , predfram = pdp_per_window.roll_predict(tstart, tend, X, toffset, T, lor, y)

#%%

pd.infer_freq(X.index)
#%%
pd.TimedeltaIndex(X.index)

#%%
pd.DateOffset(hours = pd.TimedeltaIndex(X.index))
