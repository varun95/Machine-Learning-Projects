
# coding: utf-8

# In[8]:

import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, preprocessing, svm
from sklearn.preprocessing import StandardScaler, Normalizer
import math
import matplotlib
import seaborn as sns


# In[10]:

df=pd.read_csv('home\parul\Documents\Project\autos.c')


# In[ ]:

print(df.head())


# In[ ]:

print(df.info())


# In[ ]:

print(df.describe())


# In[ ]:

df.drop(['seller', 'offerType', 'abtest', 'dateCrawled', 'nrOfPictures', 'lastSeen', 'postalCode', 'dateCreated'], axis='columns', inplace=True)


# In[ ]:

print("New Car: %d" % df.loc[df.yearOfRegistration >= 2017].count()['name'])
print("Old Car: %d" % df.loc[df.yearOfRegistration < 1980].count()['name'])
print("Cheap: %d" % df.loc[df.price < 1000].count()['name'])
print("Expensive: " , df.loc[df.price > 150000].count()['name'])
print("Low km: " , df.loc[df.kilometer < 5000].count()['name'])
print("High km: " , df.loc[df.kilometer > 200000].count()['name'])
print("Low PS: " , df.loc[df.powerPS < 40].count()['name'])
print("High PS: " , df.loc[df.powerPS > 350].count()['name'])
print("Fuel Types: " , df['fuelType'].unique())
print("Damages: " , df['notRepairedDamage'].unique())
print("Vehicle types: " , df['vehicleType'].unique())
print("Brands: " , df['brand'].unique())


# In[ ]:

#DATA CLEANING

#DROPPING DUPLICATES

dups = df.drop_duplicates(['name','price','vehicleType','yearOfRegistration'
                         ,'gearbox','powerPS','model','kilometer','monthOfRegistration','fuelType'
                         ,'notRepairedDamage'])


# In[ ]:

#TOTAL NULL VALUES

dups.isnull().sum()


# In[ ]:

dups = df.drop_duplicates(['name','price','vehicleType','yearOfRegistration'
                         ,'gearbox','powerPS','model','kilometer','monthOfRegistration','fuelType'
                         ,'notRepairedDamage'])


# In[ ]:

dups = dups[
        (dups.yearOfRegistration <= 2016) 
      & (dups.yearOfRegistration >= 1950) 
      & (dups.price >= 100) 
      & (dups.price <= 150000) 
      & (dups.powerPS >= 10) 
      & (dups.powerPS <= 500)]

print("*****************************\nData kept for analisys: %d percent of the entire Dataset\n******************************" % (100 * dups['name'].count() / df['name'].count()))


# In[ ]:

#CHECKING ALL NULL VALUES DROPPED

dups.isnull().sum()


# In[ ]:

dups['notRepairedDamage'].fillna(value='not-declared', inplace=True)
dups['fuelType'].fillna(value='not-declared', inplace=True)
dups['gearbox'].fillna(value='not-declared', inplace=True)
dups['vehicleType'].fillna(value='not-declared', inplace=True)
dups['model'].fillna(value='not-declared', inplace=True)


# In[ ]:

dups.isnull().sum()


# In[ ]:

#VISUALIZATION
categories = ['gearbox', 'model', 'brand', 'vehicleType', 'fuelType', 'notRepairedDamage']

for i, c in enumerate(categories):
    v = dups[c].unique()
    
    g = dups.groupby(by=c)[c].count().sort_values(ascending=False)
    r = range(min(len(v), 5))

    print( g.head())
    plt.figure(figsize=(5,3))
    plt.bar(r, g.head()) 
    plt.xticks(r, g.index)
    plt.show()


# In[ ]:

dups['namelen'] = [min(70, len(n)) for n in dups['name']]

ax = sns.jointplot(x='namelen', 
                   y='price',
                   data=dups[['namelen','price']], 
                    alpha=0.1, 
                    size=8)
plt.show()


# In[ ]:

labels = ['name', 'gearbox', 'notRepairedDamage', 'model', 'brand', 'fuelType', 'vehicleType']
les = {}

for l in labels:
    les[l] = preprocessing.LabelEncoder()
    les[l].fit(dups[l])
    tr = les[l].transform(dups[l]) 
    dups.loc[:, l + '_feat'] = pd.Series(tr, index=dups.index)

labeled = dups[ ['price'
                        ,'yearOfRegistration'
                        ,'powerPS'
                        ,'kilometer'
                        ,'monthOfRegistration'
                        , 'namelen'] 
                    + [x+"_feat" for x in labels]]


# In[ ]:

len(labeled['name_feat'].unique()) / len(labeled['name_feat'])


# In[ ]:

#CORRELATION
#labeled.corr()
labeled.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]


# In[ ]:

Y = labeled['price']
X = labeled.drop(['price'], axis='columns', inplace=False)


matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"1. Before":Y, "2. After":np.log1p(Y)})
prices.hist()

Y = np.log1p(Y)
plt.show()


# In[ ]:

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, Lasso, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score, train_test_split

def cv_rmse(model, x, y):
    r = np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv = 5))
    return r

# Percent of the X array to use as training set. This implies that the rest will be test set
test_size = .33

#Split into train and validation
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=test_size, random_state = 3)
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

r = range(2003, 2017)
km_year = 10000


# In[ ]:

#RANDOM FOREST USING GRID SEARCH

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

rf = RandomForestRegressor()

param_grid = { "criterion" : ["mse"]
              , "min_samples_leaf" : [3]
              , "min_samples_split" : [3]
              , "max_depth": [10]
              , "n_estimators": [500]}

gs = GridSearchCV(estimator=rf, param_grid=param_grid, cv=2, n_jobs=-1, verbose=1)
gs = gs.fit(X_train, y_train)


# In[ ]:

bp = gs.best_params_
forest = RandomForestRegressor(criterion=bp['criterion'],
                              min_samples_leaf=bp['min_samples_leaf'],
                              min_samples_split=bp['min_samples_split'],
                              max_depth=bp['max_depth'],
                              n_estimators=bp['n_estimators'])
forest.fit(X_train, y_train)
# Explained variance score: 1 is perfect prediction
print('Score: %.2f' % forest.score(X_val, y_val))


# In[ ]:

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

print(X_train.columns.values)
# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center",tick_label = X_train.columns.values)
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

