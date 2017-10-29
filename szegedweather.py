### szakdoga, szeged idลjรกrรกs modell v1
#
#elkรฉpzelรฉsek:
# 1 idลsorosan kielemzem, predikciรณt adok, รฉs visszamรฉrem az adat-
#              bรกzist
# 2 
#
# 3
#
#              
###

#==============================================================================
# Ötletláda:
    # a fejléceket külön egységesen formázni
    # megvizsgálni a változókat, milyen típusú, milyen tartományú, mennyi a hiányos érték stb.. 
# lényegében az előfeldogozási lépések lerövidítése, és ábrázolása
    # a klaszteranalízissel felbontani a főbb határokat, csoportokat
    # az elmúlt időszakban felvett adatok hozzáadása a az adathoz:aggregálás? 
    
    
    
#==============================================================================
# %%
#==============================================================================
#   I első box: alapvető modulok
#==============================================================================


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# ?



#%%
#==============================================================================
#   II a beolvasás
#==============================================================================
df = pd.read_csv("U:/adatbanyaszatialkalmazasok/bázisok/szeged-weather/weatherHistory.csv",sep= ',')
features = df.columns

# Hány nap telt el összesen, és egyenletesen (óránként történt a mérés?)

#%%
#==============================================================================
#   III előfeldolgozás 
# az idő tartomány folytonossága
# indexelés az idő szerint
#==============================================================================
#%%
"""
df["Formatted Date"].isnull().sum();
type(df["Formatted Date"]);
plt.figure(figsize=(200,4));
plt.plot_date(df["Formatted Date"],df["Temperature (C)"]);
plt.savefig("U:\\adatbanyaszatialkalmazasok\project\szakdoga\szakdoga\kep1.png");

"""
#%%
print (df["Formatted Date"].dtype)
df["Formatted Date"] = pd.to_datetime(df["Formatted Date"])
print (df["Formatted Date"].dtype)

df["month"] = df["Formatted Date"].dt.month
df["day"] = df["Formatted Date"].dt.day
df["year"] = df["Formatted Date"].dt.year
df["hour"] = df["Formatted Date"].dt.hour

df.year = df.year.astype("category")
df.moth = df.month.astype("category")
df.day = df.day.astype ("category")
df.hour = df.hour.astype("category")
# az indexelés az idő szerint
df.set_index("Formatted Date",inplace = True);

#%%
   # nem kéne kódolni a kategorikus értékeket? 
   # illetve megnézni mik ezek a kategorikus értékek?
df.Summary.value_counts();
df["Daily Summary"].value_counts();

# df["Summary"].corr(df["Daily Summary"])
df.Summary.value_counts().sum();
df.Summary = df.Summary.astype("category");

# mindenesetre eldobom a Daily Summery-t
Daily_summary = pd.Series(df["Daily Summary"]);
df = df.drop(["Daily Summary"], axis=1);

# szerkesztés
df["Precip Type"] = df["Precip Type"].astype("category");
df["Precip Type"] = df["Precip Type"].cat.codes
df = df.drop(["Loud Cover"], axis=1) 

#%% normalizálás és standardizálás


#%%

#==============================================================================
#  IV elsőnek, egy egyszerű előrejelzés a "Summary" változóra 
#==============================================================================
        
#from sklearn import SVM.svc
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, train_test_split

#split
y = pd.Series(df.Summary.cat.codes)
y.astype("category")
X = df.drop(["Summary"], axis=1)


"""     
ezek most nem kellenek
        #előbb egy split, ezeken a halmazokon fogom validálni
        def split(dtfrm,splitrate):
            lenght = len(dtfrm)
                return dtfrm[length*splitrate,:], dtfrm[length-length*splitrate,:]
        
        X_t, Xvalid, y_t, yvalid = train_test_split(X,y,test_size=0.33, random_state=0)
        
        # majd egy idősoros split
        # ez úgy működik hogy, visszaadja train és teszt indexeit, minden szeletben a teszt indexek növekednek,
        
         #ez szar valamiért nem működik, hosszú debuggolásra volna szükség..., 
         # az első tippem hogy az index formátumával van baj
            #for tr_ind, test_ind in TimeSeriesSplit(n_splits=60).split(X):
            #    print("train",tr_ind, "Test", test_ind)
            #    Xtrain, Xtest = X[tr_ind], X[test_ind]
            #    ytrain, ytest = y[tr_ind], y[test_ind]
        # ... vagy hogy magam írjam az indexelő függvényt be
        # ebből még sok bőrt le lehtne húzni
        
        # megoldás 2 : az időváltozót szétbontom, és kategorikusan kezelem, úgymint dayoftheweek,
        # weekofthemonth,year, stb... ezután az nincs szükség a dátumszerinti indexelésre
        
        # megoldás 3 vagy ötlet: készítek egy cross validátort, és a visszaméréskor csak az időben következő, azonos méretű blokkon mérek vissza.
        # de ez ua mint az 1. megoldás
        
        
         
 """
yvalid = y[pd.Timestamp('2015-01-01'):]
yvalid = yvalid.astype("category")
Xvalid = X[pd.Timestamp('2015-01-01'):]
ytrain = y[:pd.Timestamp('2015-01-01') - pd.DateOffset(day=1)]
ytrain = ytrain.astype("category")
Xtrain = X[:pd.Timestamp('2015-01-01') - pd.DateOffset(day=1)]



#%%
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import pipeline
from sklearn.ensemble import GradientBoostingClassifier

np.random.seed(1)


dt = DecisionTreeClassifier()
rfc = RandomForestClassifier()
knn = KMeans(n_clusters = 27, init = "k-means++")
gbc = GradientBoostingClassifier ( )


rfc.fit(Xtrain,ytrain)
dt.fit(Xtrain,ytrain)
knn.fit(Xtrain, ytrain)
gbc.fit(Xtrain,ytrain)

ypred_knn = knn.predict(Xvalid)
ypred_dt = dt.predict(Xvalid)
ypred_rfc = rfc.predict(Xvalid)
ypred_gbc = gbc.predict(Xvalid)


#%%
#==============================================================================
# Visszamérés
#==============================================================================


from sklearn.metrics import accuracy_score
acc_knn = accuracy_score(yvalid, ypred_knn)
acc_dt = accuracy_score(yvalid, ypred_dt)
acc_rfc = accuracy_score(yvalid, ypred_rfc)
acc_gbc = accuracy_score(yvalid, ypred_gbc)


pred_rfc=pd.Series(rfc.predict(Xvalid), index = Xvalid.index)

#%%

acc_knn_cat = accuracy_score(yvalid, ypred_knn)
acc_dt_cat = accuracy_score(yvalid, ypred_dt)
acc_rfc_cat = accuracy_score(yvalid, ypred_rfc)
acc_gbc_cat = accuracy_score(yvalid, ypred_gbc)
               

#%%

#==============================================================================
# Dependecies Plot, Dependenciesnetwork int time!
#==============================================================================
from sklearn.ensemble import partial_dependence

feature_names = Xtrain.columns
feature_numbers= range(len(feature_names))


#a 25 = 26-1 ami az y számossága eggyel csökkentve az 0tól kezdődő számossága miatt
label = len(ytrain.value_counts())
pdp_fig, pdp_axs = partial_dependence.plot_partial_dependence(gbc,Xtrain,feature_numbers,feature_names, label)
plt.subplots_adjust(top=1.5)  # tight_layout causes overlap with suptitle



#%%
from mpl_toolkits.mplot3d import Axes3D

print('Custom 3d plot via ``partial_dependence``')
fig = plt.figure()

target_feature = (3,7)
pdp, axes = partial_dependence.partial_dependence(gbc, target_feature,X=Xtrain, grid_resolution=100, )
XX, YY = np.meshgrid(axes[0], axes[1])
Z = pdp[0].reshape(list(map(np.size, axes))).T
ax = Axes3D(fig)
surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1,
                       cmap=plt.cm.BuPu, edgecolor='k')
ax.set_xlabel(feature_names[target_feature[0]])
ax.set_ylabel(feature_names[target_feature[1]])
ax.set_zlabel('Partial dependence')
#  pretty init view
ax.view_init(elev=22, azim=122)
plt.colorbar(surf)
plt.subplots_adjust(top=0.9)

plt.show()



#%%
feature_importances = pd.Series(gbc.feature_importances_, Xtrain.columns)
feature_plot = feature_importances.plot(kind="bar")


