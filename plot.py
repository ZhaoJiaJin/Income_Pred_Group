#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

trainfile = "data/tcd-ml-1920-group-income-train.csv"

ff = lambda x : (x.split(" ")[0])

traindata = pd.read_csv(trainfile,na_values=["#NUM!"],dtype={'Work Experience in Current Job [years]': float, 'Yearly Income in addition to Salary (e.g. Rental Income)':float}, converters={'Yearly Income in addition to Salary (e.g. Rental Income)':ff})

category_cols = ['Housing Situation','Satisfation with employer','Gender','Country', 'Profession', 'University Degree', 'Hair Color']

headers = traindata.columns.values
#headers = ["Age"]
print(traindata.dtypes)

missing = traindata.isnull().sum()
totalcnt = np.product(traindata.shape)
misscnt = missing.sum()
print("total cell:{0}, missing cell:{1}, percentage:{2}".format(totalcnt,misscnt,misscnt / totalcnt * 100))

print("Missing percentage per columns:", np.around(traindata.isnull().mean()*100, decimals = 2))

for col in category_cols:
    print('variable: ', col, ' number of labels: ', traindata[col].nunique())

print('total records: ', len(traindata))

total_records = len(traindata)
sys.exit(1)

#for col in category_cols:
#    temp_df = pd.Series(traindata[col].value_counts() / total_records)
#    fig = temp_df.sort_values(ascending=False).plot.bar()
#    fig.set_xlabel(col)
#    fig.set_ylabel('Percentage of records')
#    #plt.show()
#    plt.savefig(col)


for col in headers:
    if col not in category_cols and col != "Total Yearly Income [EUR]":
        print(col)
        #plt.scatter(traindata[col], traindata["Total Yearly Income [EUR]"], color= 'red')
        #plt.title('{0} Vs. Income in EUR'.format(col))
        #plt.ylabel('Income in EUR', fontsize=12)
        #plt.xlabel(col, fontsize=12)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(traindata[col], traindata["Total Yearly Income [EUR]"], color= 'red')
        ax.set_xlabel(col)
        ax.set_ylabel('Income')
        ax.set_title('{0} Vs. Income in EUR'.format(col))
        fig.savefig('{0}Vs. Income in EUR.png'.format(col))
        #plt.savefig('{0}Vs. Income in EUR.png'.format(col))
