# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 21:01:03 2017

ebben egy idősoros pipeline-t kéne megvalósítani,
azaz, ami lépéseket a különböző adathalmazokon ua kell csinálni azt egy itt
létrehozott pipeline tegye
pl : fit, score,split és cross-validation, tunning?, featureselection?, 
standardize&normalize,és indexelés a dátum szerint majd felvenni a különböző i-
dőegységeket(s,min,hour, day, etc..)





@author: User
"""
import sklearn.pipeline as ppl
pipe = ppl.make_pipeline
uni = ppl.make_union


