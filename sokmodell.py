# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 20:12:48 2017


Sok modellt egyszerre kiértékelő függvény és script
 mindenféle problémára, egy vegyes saláta a cél, a modellek erősségének gyors
 összehasonlítására

 esetleg ábrázolni az eredményeket,
 vagy ensamble- technikával összerakni
 


@author: User
"""
"""
#==============================================================================
# OKOSKODÁSAIM:
    
MInden modellnek lenne egy bemeneti adahalmaza. Ezt saját tulajdonságai alapján
az eredeti halmazból feature-selection módszerekkel választódna ki. (bármit is 
jelentsen ez...) 
A ezután jönnénke a pipleine-ra már gyakrabban használt lépések, a skálázás, és 
modellezés, visszamérés.
Természetesen ehhez szükésg volna  a modellek helyes és pontos ismeretére..
Végül ezeket összegezni volna hasznos.    


    
#==============================================================================



from sklearn import 



def preprocess(df):
    N, p = df.shape()
    cols = df.columns()
    for col in cols:
 #       milyen változó?
 # hiányosságok 
 # szórása?
 # 
 
 def bestemodell ():
     modells = [];
     