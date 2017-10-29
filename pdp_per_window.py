# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 08:02:32 2017

@author: Dániel
"""



import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from sklearn.ensemble import  GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
 

def my_pdp_plotter(X,model):
    print("Xek start")
    szam = 0
    kvt = {}
    lap = {}
    fig = plt.figure(figsize=(10,14))

    for col in X.columns:
        # oszlopcsoport  
        #  lap = {}
        print("start",col)
        iterat = 0
        kvt["X_"+str(col)] = lap
        x = []
        y = []
      
        for i in X[col].value_counts().index.sort_values():
            
            # kell egy új adathalmaz (vajon a copy aleggyorsabb erre?)
            Xcopy = X.copy() 
            
          #  Xcopy = Xcopy.sort_values(by = col,ascending = True)
            # átállítom az adathalmaz megfelelő oszlop nem i értékeit, i-re
            Xcopy.loc[Xcopy.loc[:, col] != i,col] = i 
            # átlagolással hozzárendelem az adott adathalmaz predikcióit
            pre = np.mean(model.predict(Xcopy))
            # egy dictbe rendezem az adathalmazokat 
            #lap["X_"+str(iterat)] = Xcopy, pre

            x.append(i)
            y.append(pre)

            szam = szam+1
       # az adathalmazokat oszloponként rendezem dict-be
       # kvt["X_"+str(col)] = lap
        # ábra 
        plt.subplot(100*X.shape[1]+10+len(kvt))
        if X[col].dtype is np.dtype(np.int8):
            plt.bar(x,y)
        else:
            plt.plot(x,y, marker='o', markeredgecolor='k',markerfacecolor='b')
    
        plt.xlabel(col)
        plt.ylabel("AvePred")
        plt.subplots_adjust(hspace=0.8)
         
    return fig
     




class datawindow:
    
    def __init__(self, starttime, endtime, dataset):
        try:
            self.starttime = pd.Timestamp(starttime)
            self.endtime = pd.Timestamp(endtime)
        except:
            raise ValueError ( " can'r read Timestamp type " )
   
        # If FALSE -> like autamatic stand out the freq trougth pd.infer_freq() method
        # set the step timespan in the sampling_time variable
        if pd.infer_freq(dataset.index) == None:
            self.sampling_time = pd.DateOffset(hours=1)
            print( " Sampling time is default 1 hour " )
        else:
            self.sampling_time = pd.DateOffset(hours = pd.TimedeltaIndex(dataset.index))
            print( " sampling time is frequency of the data " )      
       
        #throught this timerange variable can we reach the data. 
        # All function uses with loc[] (?) 
        self.zeroth_timerange = pd.date_range(start = self.starttime, end = self.endtime, freq = self.sampling_time)
        self.timerange = self.zeroth_timerange
        
        self.dataset  = dataset.copy()
    
    #to see what is on the window 
    def see(self):
            return self.dataset.loc[self.timerange]
        
    
    # this function is might to make a new start and endtime with shifting one samplig step on the dataset
    # do not yileds a new datawindow (yet) !!!!
    # not exactly the watied results, cant take the step back, or run again
    """ Question: after a reindex come out a new window object,and model, or the window self will be change?
    Answer: reindex just changes the timerange. You can count the reindexing, and after that follow modifies. The 
    mappng the data in the timerange was not writed here
    """
    def reindex(self):
        self.starttime = self.starttime + self.sampling_time
        self.endtime = self.endtime + self.sampling_time
        self.timerange = pd.date_range(start = self.starttime, end = self.endtime, freq = self.sampling_time)
        return self
    
    # don't work, don't know why..
    def default_index(self,starttime, endtime):
        try:
            self.starttime = pd.Timestamp(starttime)
            self.endtime = pd.Timestamp(endtime)
        except:
            raise ValueError ("can'r read Timestamp type")
    
    # making reindexing beetven T end endtime, step by step
    # return a list of timeranges in DatetimeIndex ( times) 
    # if T is less or equals self.endtime, then it comes back with Valuerror
    # !!!  for loop goes until endtime == T (timerange do not contain T )
    def reindex_til_T(self,T):
        print("Rolling...\n")
        if(T <= self.endtime):
            raise ValueError("lekváár, T<end\n")
        times =  []
        timevec = pd.date_range(start=self.endtime , end= pd.Timestamp(T-self.sampling_time)  , freq=self.sampling_time)
        for self.endtime in timevec:
            times.append(self.timerange)  
            self.reindex()
        return times
    
    # if we want to run the window on the full dataset
    def reindex_til_end(self):
        print("Rolling...\n")
        if(self.dataset.index.max() <= self.endtime):
            raise ValueError("lekváár, T<end\n")
        times =  []
        timevec = pd.date_range(start = self.endtime , end = self.dataset.index.max(), freq = self.sampling_time)
        for self.endtime in timevec:
            self.reindex()
            times.append(self.timerange)#??   
        return times
        
    # using a generator function t create the timeranges
    # under construction...
    def generator_roll(self,T):
        for t in pd.date_range(start=self.endtime, end = T, freq=self.sampling_time):
            yield 
        return t 
    

def roll_predict(start,end,data,tforward,T,model, target):
    # build up the windows, and the time sequencies
    Xw = datawindow(start,end,data)
    yw = datawindow(start,end,target)
    timeframes = Xw.reindex_til_T(T)
    pics =[]
    
    # for the output, creating a Df
    rolled = pd.DataFrame(columns = ['t','yTpred','ypred','y','metriced score'])
    print("Length of timerangesvector: ",len(timeframes))
    for rng in timeframes:
        print( "Y:",yw.dataset.loc[rng],"\n\n yw(t+tforw):",yw.dataset.loc[rng.max() + tforward],"\n-----\n")
        print("range:",rng.min(),"--",rng.max())
        
      # print(Xw.dataset.loc[rng,:].shape, yw.dataset.loc[rng].shape)
        
        model.fit( Xw.dataset.loc[rng,], yw.dataset.loc[rng] );
        pics.append(my_pdp_plotter(Xw.dataset.loc[rng,],model)   )
        
        plt.title(rng.max())
        plt.show()
     
        rolled = rolled.append({
                        't': rng.max()+ tforward,
                        'yTpred': model.predict(Xw.dataset.loc[T].reshape(1,-1)),
                        'ypred': model.predict(Xw.dataset.loc[rng.max() + tforward].reshape(1,-1)),
                        'y': yw.dataset.loc[rng.max()+tforward]
                                }, ignore_index= True)
    
                                    
    return pics, rolled

        


if __name__ == "__main__" :

# adatok beolvasása és előfeldolgozása      
    df = pd.read_csv("U:/adatbanyaszatialkalmazasok/bázisok/szeged-weather/weatherHistory.csv",sep= ',')
    df.head()
    df = df.sort_values(by = "Formatted Date")
##ez itten faszság, nem működik
   df["Formatted Date"] = pd.DatetimeIndex(df["Formatted Date"].values, freq= pd.infer_freq(df["Formatted Date"]))
    
    
    df.set_index("Formatted Date",inplace = True,)
    
    
    """
    kód arról, hogy mesterségesen előállítom a DatetimeIndex-t, de sajna nem az igazi, mert a Formatted Date redundáns, és hibás
    """
    # df.index = pd.date_range(start=df.index.min(), end=df.index.max(), freq = "H")
    # df["Formatted Date"].value_counts()
    # df["Formatted Date"] = df["Formatted Date"].drop_duplicates() 
        
    df.Summary = df.Summary.astype("category");
    df.Summary = df.Summary.cat.codes
    
    df = df.drop(["Daily Summary"], axis=1);
    df = df.drop(["Loud Cover"], axis=1) 
    
    df["Precip_Type"] = df["Precip Type"].astype("category");
    df.Precip_Type = df["Precip_Type"].cat.codes
    
    df = df.drop(["Precip Type"], axis=1)
    
    df["Pressure (millibars)"] = df["Pressure (millibars)"].astype("category");
    
   #%%
   
   
 # modellek ill. taníntó- és tesztelő halmazok    
    gbr = GradientBoostingRegressor()
    rf = RandomForestRegressor()
    
    X= df["2006"].copy()
    X = X.drop(["Humidity"],axis = 1)
    y = df["2006"].Humidity
    
    rf.fit(X,y)
    gbr.fit(X,y)
    
    
    y.describe(), X.describe(include="all")
 #%% 
# próbálgatások
    
    f = my_pdp_plotter(X.head(10),gbr)
    
# teszt környezet
    tstart = "2006-01-10 10"
    tend = "2006-01-10 17"
    tforward = 48
    T = pd.Timestamp("2006-01-10 23")
    
    pdpics, predictframe = roll_predict(tstart, tend, X, tforward, T, gbr, y)    
    
    dw = datawindow(tstart,tend,X)
    dw.dataset.loc[T].reshape(1,-1)