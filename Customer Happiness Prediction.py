# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 16:11:31 2017

@author: Varun
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, make_scorer

train = pd.read_csv("C:/Users/Varun/Desktop/New folder (7)/train.csv")
test = pd.read_csv("C:/Users/Varun/Desktop/New folder (7)/test.csv")

print(train.head())

print(test.head())

print(test.describe())

print(test.info())
print(train.info())

#Function to clean the data

stops = set(stopwords.words("english"))
def cleanData(text, lowercase = False, remove_stops = False, stemming = False):
    tx = str(text)
    tx = re.sub(r'[^A-Za-z0-9\s]',r'',tx)
    tx = re.sub(r'\n',r' ',tx)
    
    if lowercase:
        tx = " ".join([w.lower() for w in tx.split()])
        
    if remove_stops:
        tx = " ".join([w for w in tx.split() if w not in stops])
    
    if stemming:
        st = PorterStemmer()
        tx = " ".join([st.stem(w) for w in tx.split()])

    return tx

#Joining both the data
test['Is_Response'] = np.nan
alldata = pd.concat([train, test]).reset_index(drop=True)

#Cleaning the Description in New Combined Dataset
alldata['Description'] = alldata['Description'].map(lambda x: cleanData(x, lowercase=True, remove_stops=True, stemming=True))

#initialise the functions - we'll create separate models for each type.
countvec = CountVectorizer(analyzer='word', ngram_range = (1,1), min_df=150, max_features=500)
tfidfvec = TfidfVectorizer(analyzer='word', ngram_range = (1,1), min_df = 150, max_features=500)


#Feature Creation
bagofwords = countvec.fit_transform(alldata['Description'])
tfidfdata = tfidfvec.fit_transform(alldata['Description'])

#Label Encoding
cols = ['Browser_Used','Device_Used']

for x in cols:
    lbl = LabelEncoder()
    alldata[x] = lbl.fit_transform(alldata[x])
    
# create dataframe for features
bow_df = pd.DataFrame(bagofwords.todense())
tfidf_df = pd.DataFrame(tfidfdata.todense())


# set column names
bow_df.columns = ['col'+ str(x) for x in bow_df.columns]
tfidf_df.columns = ['col' + str(x) for x in tfidf_df.columns]


# create separate data frame for bag of words and tf-idf

bow_df_train = bow_df[:len(train)]
bow_df_test = bow_df[len(train):]

tfid_df_train = tfidf_df[:len(train)]
tfid_df_test = tfidf_df[len(train):]


# split the merged data file into train and test respectively
train_feats = alldata[~pd.isnull(alldata.Is_Response)]
test_feats = alldata[pd.isnull(alldata.Is_Response)]


### set target variable

train_feats['Is_Response'] = [1 if x == 'happy' else 0 for x in train_feats['Is_Response']]


# merge count (bag of word) features into train
train_feats1 = pd.concat([train_feats[cols], bow_df_train], axis = 1)
test_feats1 = pd.concat([test_feats[cols], bow_df_test], axis=1)

test_feats1.reset_index(drop=True, inplace=True)

# merge into a new data frame with tf-idf features
train_feats2 = pd.concat([train_feats[cols], tfid_df_train], axis=1)
test_feats2 = pd.concat([test_feats[cols], tfid_df_test], axis=1)


mod1 = GaussianNB()
target = train_feats['Is_Response']

#NAIVE BAYES
## Naive Bayes 1
print(cross_val_score(mod1, train_feats1, target, cv=5, scoring=make_scorer(accuracy_score)))

## Naive Bayes 2 
print(cross_val_score(mod1, train_feats2, target, cv=5, scoring=make_scorer(accuracy_score)))

#First set of predictions

clf1 = GaussianNB()
clf1.fit(train_feats1, target)

clf2 = GaussianNB()
clf2.fit(train_feats2, target)

preds1 = clf1.predict(test_feats1)
preds2 = clf2.predict(test_feats2)



def to_labels(x):
    if x == 1:
        return "happy"
    return "not_happy"

sub1 = pd.DataFrame({'User_ID':test.User_ID, 'Is_Response':preds1})
sub1['Is_Response'] = sub1['Is_Response'].map(lambda x: to_labels(x))

sub2 = pd.DataFrame({'User_ID':test.User_ID, 'Is_Response':preds2})
sub2['Is_Response'] = sub2['Is_Response'].map(lambda x: to_labels(x))

sub1 = sub1[['User_ID', 'Is_Response']]
sub2 = sub2[['User_ID', 'Is_Response']]

# Writing Result files
sub1.to_csv('C:/Users/Varun/Desktop/New folder (7)/sub1c.csv', index=False)
sub2.to_csv('C:/Users/Varun/Desktop/New folder (7)/sub2tf.csv', index=False)


#LIGHT GBM

# Setting the data in format lgb accepts
d_train = lgb.Dataset(train_feats1, label = target)

#Setting up the Parameters
params = {'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_error',
    'learning_rate': 0.05, 
    'max_depth': 7, 
    'num_leaves': 21, 
    'feature_fraction': 0.3, 
    'bagging_fraction': 0.8, 
    'bagging_freq': 5}
lgb_cv = lgb.cv(params, d_train, num_boost_round=500, nfold= 5, shuffle=True, stratified=True, verbose_eval=20, early_stopping_rounds=40)
nround = lgb_cv['binary_error-mean'].index(np.min(lgb_cv['binary_error-mean']))

## Training the model
model = lgb.train(params, d_train, num_boost_round=nround)

#Making Predictions
preds = model.predict(test_feats1)
print(preds)

# Writing the Result files

def t_labels(x):
    if x > 0.66: 
        return "happy"
    return "not_happy"
    
sub3 = pd.DataFrame({'User_ID':test.User_ID, 'Is_Response':preds})
sub3['Is_Response'] = sub3['Is_Response'].map(lambda x:t_labels(x))
sub3 = sub3[['User_ID','Is_Response']]
sub3.to_csv('C:/Users/Varun/Desktop/New folder (7)/subcat.csv', index=False)

   

