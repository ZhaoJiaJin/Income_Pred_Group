#!/usr/bin/env python
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
from sklearn.base import TransformerMixin
import numpy
import pandas as pd
from sklearn import datasets, linear_model, preprocessing
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import sys
import lightgbm as lgb

#import affinity
#import multiprocessing
#affinity.set_process_affinity_mask(0, 2**multiprocessing.cpu_count()-1)

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == numpy.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)



ff = lambda x : (float(x.split(" ")[0]))

# Load the diabetes dataset
#alldata = np.loadtxt("train.csv",delimiter=",", usecols=range(0,2))
alldata = pd.read_csv("data/tcd-ml-1920-group-income-train.csv",na_values=["#NUM!","nA"],converters={'Yearly Income in addition to Salary (e.g. Rental Income)':ff})
predictdata = pd.read_csv("data/tcd-ml-1920-group-income-test.csv",na_values=["#NUM!","nA"],converters={'Yearly Income in addition to Salary (e.g. Rental Income)':ff})
#alldata = pd.read_csv("train.csv")
#predictdata = pd.read_csv("predict.csv")


# rename
newname = {
    "Year of Record":"Year",
    "Housing Situation":"Housing",
    "Crime Level in the City of Employement":"Crime",
    "Work Experience in Current Job [years]":"Experience",
    "Satisfation with employer":"Satisfation",
    "Size of City":"CitySize",
    "University Degree":"Degree",
    #Wears Glasses,Hair Color,
    "Body Height [cm]":"Height",
    "Yearly Income in addition to Salary (e.g. Rental Income)":"addition",
    "Total Yearly Income [EUR]":"income"
}

alldata = alldata.rename(columns=newname)
predictdata = predictdata.rename(columns=newname)

drop_col = ["Instance","addition","income"]

allcol = alldata.columns

feature_col = []

# drop some columns
for i in allcol:
    if i not in drop_col:
        feature_col.append(i)

print(feature_col)



#alldata = alldata.dropna(subset=allcol)

(ROW,COL) = alldata.shape
(PROW,PCOL) = predictdata.shape


# y = income - addition income
predictIdx = predictdata['Instance'].values

target = alldata['income'].values - alldata['addition'].values

predict_addition = predictdata['addition']



alldata = alldata[feature_col]
predictdata = predictdata[feature_col]

# concat train and predict data
totaldata = pd.concat([alldata,predictdata])

# then fill missing values
totaldata = DataFrameImputer().fit_transform(totaldata)


print(totaldata.head())
print(totaldata.shape)
labelcol = []
numcol = []

for (columnName, columnData) in totaldata.iteritems():
    if columnName != "Profession":
        if (columnData.dtype == object):
            labelcol.append(columnName)
        else:
            numcol.append(columnName)

print(labelcol)

for i in labelcol:
    totaldata[i] = totaldata[i].astype(str)


# perform scale on numerical columns, one for encoder for string column and ordinal ecoder for Profession
preprocess = make_column_transformer(
    (numcol, preprocessing.StandardScaler()),
    (labelcol, preprocessing.OneHotEncoder(drop='first')),
    (["Profession"],preprocessing.OrdinalEncoder()),
)

# do data preprocess
totaldata = preprocess.fit_transform(totaldata).toarray()

#print("one hot encoder")
#for i in range(0,len(strlist)):
#    #print(labeldata[:,i:i+1])
#    ldata = labeldata[:,i:i+1]
#    #ldata = ldata.astype("category")
#    feaLen = len(numpy.unique(ldata.astype(str)))
#    if feaLen < 20:
#        drop_enc = preprocessing.OneHotEncoder(drop='first',categories='auto').fit(ldata)
#        newdata = drop_enc.transform(ldata).toarray()
#
#        newX = numpy.append(newX, newdata[:ROW, :], axis=1)
#        newPX = numpy.append(newPX, newdata[ROW:,:], axis = 1)
#    else:
#        enc = preprocessing.OrdinalEncoder().fit(ldata)
#        newdata = enc.transform(ldata)
#
#        newX = numpy.append(newX, newdata[:ROW, :], axis=1)
#        newPX = numpy.append(newPX, newdata[ROW:,:], axis = 1)





X = totaldata[:ROW,:]
predictX = totaldata[ROW:,:]



Y = target.astype(float)
Y = numpy.log(Y)

print(X[0:1,:])
print(Y[0:35])

y = Y

(XROW,XCOL) = X.shape

train_features = X
train_labels = y

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)

## Instantiate model with 1000 decision trees
#rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
## Train the model on training data
#rf.fit(train_features, train_labels);
#
#predictRes = rf.predict(predictX)

X_train, X_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.2, random_state=0)
## Create linear regression object
#regr = linear_model.LinearRegression()
## Train the model using the training sets
#regr.fit(X_train, y_train)
## Make predictions using the testing set
#predict_y = regr.predict(X_test)

params = {
          'max_depth': 20,
          #'num_leaves': 2**20 / 5,
          'learning_rate': 0.001,
          "boosting": "gbdt",
          "bagging_seed": 11,
          "metric": 'mse',
          "verbosity": -1,
         }
trn_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_test, label=y_test)
# test_data = lgb.Dataset(X_test)
# train model
clf = lgb.train(params, trn_data, 50000, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds=500)
pre_test_lgb = clf.predict(X_test)

# calcute mse
predict_y = clf.predict(X_test)

merr =  mean_squared_error(y_test, predict_y)

# predict
predictRes = clf.predict(predictX)

predictRes = numpy.exp(predictRes) + predict_addition


with open("resultl2_lgb2.csv","w") as f:
    index = predictIdx
    f.write("Mean squared error: %.2f\n" % merr)
    f.write("Instance,Total Yearly Income [EUR]\n")
    (pre_data_row,pre_data_col) = predictdata.shape
    for i in range(0, pre_data_row):
        f.write("%d,%d\n" % (index[i], predictRes[i]))

