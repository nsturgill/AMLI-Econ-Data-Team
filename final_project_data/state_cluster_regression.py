# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 19:14:13 2019

@author: Natal
"""
import pandas as pd
from sklearn.preprocessing import scale
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

data = pd.read_csv("Dropbox/AMLI/final_project_data/full_select_feature_dat.csv", index_col=0)
descdat = pd.DataFrame()


for state, df_state in data.groupby('full_state_cluster'):

    x = (df_state.iloc[:, 17:37])
    y = df_state['Estimate; INCOME AND BENEFITS (IN 2017 INFLATION-ADJUSTED DOLLARS) - Total households - Median household income (dollars)']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 2)
    
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)
    rsquared = round(lin_reg.score(x_train,y_train),4)
    maxcoef = round(max(lin_reg.coef_), 4)
    inter = round(lin_reg.intercept_, 4)
    y_pred = lin_reg.predict(x_test)
    mse = round(math.sqrt(mean_squared_error(y_test, y_pred)), 4)
    mae = round(mean_absolute_error(y_test, y_pred), 4)
    
    
    
    desclist = pd.Series()
    y_desc = y.describe()
    collist = pd.concat([pd.Series(["rsquared", "maxcoef", "inter", "mse", "mae"]), pd.Series(y_desc.index)])
    desclist = desclist.append(pd.Series([rsquared, maxcoef, inter, mse, mae]))
    desclist = desclist.append(pd.Series(y_desc.values))
    desclist = list(desclist)
    descdat[state] = desclist

    
    #print(y.describe())
    print("\nStatistics for: ", state)
    print("R Squared: ", rsquared)
    print("Max Coefficient: ", maxcoef , "\nIntercept: ", inter)
    print("MSE: ", mse)
    print("MAE: ", mae)

 

descdat.index = pd.concat([pd.Series(["rsquared", "maxcoef", "inter", "mse", "mae"]), pd.Series(y_desc.index)])

descdat = descdat.T


descdat.to_csv(path_or_buf = "Dropbox/AMLI/final_project_data/state_cluster_regression_results.csv")





