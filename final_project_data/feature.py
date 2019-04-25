# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 23:32:41 2019

@author: Natal
"""
import pandas as pd
from sklearn.feature_selection import RFE
import statsmodels.api as sm
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


data = pd.read_csv("Dropbox/AMLI/final_project_data/cluster_combined_dat.csv")


x = (data.iloc[:, 6:50])
y = data['Estimate; INCOME AND BENEFITS (IN 2017 INFLATION-ADJUSTED DOLLARS) - Total households - Median household income (dollars)']

#perform regression
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)


featurecols = x.columns
rfe = RFE(lin_reg)
fit = rfe.fit(x,y)
#Number of features selected: 21 
print("Num of Features: " + str(fit.n_features_))
rfelist = pd.DataFrame(list(zip(featurecols,fit.support_)))
rfelist = rfelist[rfelist[1] == True]

featurelist = rfelist











model = sm.OLS(y, x).fit()
model.bse
feat = pd.DataFrame(list(zip(model.bse, x.columns)))
feat = feat.nlargest(22, 0)
featurelist = feat.merge(featurelist, left_on = 1, right_on = 0)




f_reg = pd.DataFrame(f_regression(x,y))
f_reg = f_reg.T
f_reg["features"] = x.columns

featurelist = f_reg.merge(featurelist, left_on = "features", right_on = 1)
features = featurelist["features"]

cluster = (data.iloc[:, 50:63])
locdat = (data.iloc[:, 1:6])
finaldat = data[features] 
finaldat = pd.concat([cluster, locdat, finaldat], axis = 1)
finaldat.to_csv(path_or_buf = "Dropbox/AMLI/final_project_data/full_select_feature_dat.csv")
 