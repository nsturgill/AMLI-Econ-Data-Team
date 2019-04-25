# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 19:08:40 2019

@author: Natal
"""

# %% In [1]:
import numpy as np
import sklearn as sk
import pandas as pd
import csv



# %%In [2] 
 
#file = open("Desktop/final_project_data/ACS_17_5YR_DP02_with_ann.csv")
#file = csv.reader(file, delimiter=',', header = True)
#soc_census_dat = pd.DataFrame(file)

datsoc = pd.read_csv("Desktop/final_project_data/ACS_17_5YR_DP02_with_ann.csv", header = 0) #Read in Social Factors Data

# %% In[3]
soc_census_dat = datsoc
if "GEO.id" in soc_census_dat.columns:                #getting rid of other zipcode columns
    soc_census_dat = soc_census_dat.drop(["GEO.id","GEO.id2"], axis = 1)    
zipcode  = soc_census_dat["GEO.display-label"]
if len(soc_census_dat) > 600:                        #getting rid of duplicate data, storing estimate
    testdat = soc_census_dat.iloc[:,1:]
    testdat = testdat[testdat.columns[::2]]
    testdat = testdat[testdat.columns[::2]]

soc_census_dat = testdat
#print(soc_census_dat)
soc_census_dat.columns = soc_census_dat.iloc[0,:]     #recreated column names to match full descriptive names 
#print(soc_census_dat)
temp = zipcode.str.split(" ",n = 1, expand = True)
soc_census_dat["GEO.display-label"] = temp[1]         #parsed zipcode out and renamed column

soc_census_dat.rename(columns={'GEO.display-label': 'zipcode'}, inplace=True)

soc_census_dat["zipcode"][0] = 0

phold = soc_census_dat["zipcode"]                     #changed zipcode to numeric int
phold = phold.astype(float)
soc_census_dat["zipcode"] = phold.astype(int)
#%% In[4]
dateco = pd.read_csv("Desktop/final_project_data/Census 5-yr est Selected Econ Factors.csv", header = 0) # read in econ factors data

#%% In [5]
eco_census_dat = dateco                      #see above comments 
if "GEO.id" in eco_census_dat.columns:
    eco_census_dat = eco_census_dat.drop(["GEO.id","GEO.id2"], axis = 1)    
zipcode  = eco_census_dat["GEO.display-label"]
if len(eco_census_dat) > 500:
    testdat = eco_census_dat.iloc[:,1:]
    testdat = testdat[testdat.columns[::2]]
    testdat = testdat[testdat.columns[::2]]

eco_census_dat = testdat
#print(eco_census_dat)
eco_census_dat.columns = eco_census_dat.iloc[0,:]
#print(eco_census_dat)
temp = zipcode.str.split(" ",n = 1, expand = True)
eco_census_dat["GEO.display-label"] = temp[1] 

eco_census_dat.rename(columns={'GEO.display-label': 'zipcode'}, inplace=True)

eco_census_dat["zipcode"][0] = 0

phold = eco_census_dat["zipcode"]
phold = phold.astype(float)
eco_census_dat["zipcode"] = phold.astype(int)

#%% In[6]
datage = pd.read_csv("Desktop/final_project_data/ACS_17_5YR_S0101_with_ann.csv", header = 0)   #read in age factors

#%% In [7]
age_census_dat = datage                  #see above comments 
if "GEO.id" in age_census_dat.columns:
    age_census_dat = age_census_dat.drop(["GEO.id","GEO.id2"], axis = 1)    
zipcode  = age_census_dat["GEO.display-label"]
if len(age_census_dat) > 500:
    testdat = age_census_dat.iloc[:,1:]
    testdat = testdat[testdat.columns[::2]]
    testdat = testdat[testdat.columns[::2]]

age_census_dat = testdat
#print(eco_census_dat)
age_census_dat.columns = age_census_dat.iloc[0,:]
#print(eco_census_dat)
temp = zipcode.str.split(" ",n = 1, expand = True)
age_census_dat["GEO.display-label"] = temp[1] 

age_census_dat.rename(columns={'GEO.display-label': 'zipcode'}, inplace=True)

age_census_dat["zipcode"][0] = 0

phold = age_census_dat["zipcode"]
phold = phold.astype(float)
age_census_dat["zipcode"] = phold.astype(int)

#%% In[8]:

datirs = pd.read_csv("Desktop/final_project_data/16zpallagi.csv", header = 0)   #Read in IRS data

#columns beginning with N need count and A need average
#%% IRS
irs_dat = datirs

sortdat = irs_dat.iloc[:,18:]             #pulling out Ns and As columns

sumdat = sortdat[sortdat.columns[sortdat.columns.to_series().str.contains('N')]]  #split data
avgdat = sortdat[sortdat.columns[sortdat.columns.to_series().str.contains('A')]]
sumdat["zipcode"] = irs_dat["zipcode"]
avgdat["zipcode"] = irs_dat["zipcode"]

sumdat = sumdat.groupby("zipcode").agg("sum")       #aggregate data (sum or mean) by zipcode
avgdat = avgdat.groupby("zipcode").agg("mean")

otherdat = irs_dat.iloc[:,0:18]
otherdat["zipcode"] = irs_dat["zipcode"]
otherdat = otherdat.groupby("zipcode").agg("sum")    #aggregate remaining columns (mean) by zipcode
if "agi_stub" in otherdat.columns:
    otherdat.drop(["STATEFIPS", "agi_stub"], axis = 1, inplace = True)     #create new datafram that has unique zipcode
statedat = pd.concat([irs_dat["STATE"], irs_dat["zipcode"]], axis = 1)     #as index
statedat = statedat.drop_duplicates(subset = ["zipcode"], keep='first')
statedat.set_index(statedat["zipcode"], inplace = True)
cleanirs = statedat.join(otherdat,how = "outer" )
cleanirs = cleanirs.join(sumdat, how = "outer")
cleanirs = cleanirs.join(avgdat, how = "outer")

#%% In[9]:
datzil = pd.read_csv("Desktop/final_project_data/Zip_Zhvi_Summary_AllHomes.csv", header = 0,encoding='latin-1') #readin zillow data

#%% In[10]:
zillow = datzil                      #drop what we agreed on dropping
if "PeakMonth" in zillow.columns:
    zillow.drop(["RegionID","MoM","QoQ","5Year","10Year", "PeakMonth","PeakQuarter","PeakZHVI","PctFallFromPeak","LastTimeAtCurrZHVI"], axis = 1, inplace = True)

zillow.rename(columns={'RegionName': 'zipcode'}, inplace=True)

#%% In[11]:

datbus = pd.read_csv("Desktop/final_project_data/zbp16detail.txt", header = 0,encoding='latin-1') #read in business data

#%% In[12]:
business = datbus

#business.loc[business["zip"] == '------']
#business.filter(like = "------", axis = 0)
#
#business["naics"] = business["naics"].str.rstrip()
#
#
#business.filter(like = "------", axis = 0)

#business[business['naics'] == '------']      #alternate way

business = business.drop_duplicates(subset=["zip"], keep='first') # pull instances of "------"
business.rename(columns={'zip': 'zipcode'}, inplace=True)

#%% In[13]:
cleanirs.reset_index(drop=True, inplace = True)            #reset index in order to merge all data into one dataframe

zilirsmerge = pd.merge(zillow, cleanirs, on=['zipcode'])   #default to left merge on zipcode

m1busmerge = pd.merge(zilirsmerge, business, on=['zipcode'])

m2agemerge = pd.merge(m1busmerge, age_census_dat, on=['zipcode'])

m3socmerge = pd.merge(m2agemerge, soc_census_dat, on=['zipcode'])

finaldat = pd.merge(m3socmerge, eco_census_dat, on=['zipcode'])

#%% In[14]:
if "STATE" in finaldat.columns:                                     #becuase we dont need state twice
    finaldat = finaldat.drop("STATE", axis = 1)

finaldat.iloc[:,5:] = finaldat.iloc[:,5:].apply(lambda x: pd.to_numeric(x, errors='coerce') ) #make everything numeric that should be
                                                                                # 'corece' converts non numbers to NaN

test = finaldat.isna().sum()                               #create list of na's by column
testdf = pd.DataFrame(test)
testdf.reset_index(inplace=True)
testdf.columns = ['index','totals']

droplist = testdf[(testdf['totals']> 3000)]                #remove columns with more na's than 3000 (more than 20% of data NA)
#for i in finaldat.isna().sum():
#    if i > 3000:
#        droplist.append(testdf["index"][count])
#    count = count+1
#
if len(droplist) > 0:
    finaldat = finaldat.drop(droplist['index'], axis = 1)

#%% In[15]:
    
finaldat.isna().sum()
hold = finaldat["Estimate; INCOME AND BENEFITS (IN 2017 INFLATION-ADJUSTED DOLLARS) - Total households - Median household income (dollars)"]
finaldat = finaldat.loc[:,finaldat.columns != "Estimate; INCOME AND BENEFITS (IN 2017 INFLATION-ADJUSTED DOLLARS) - Total households - Median household income (dollars)"]

vis = finaldat.corr()

# Create correlation matrix
corr_matrix = finaldat.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.7
to_drop = [column for column in upper.columns if any(upper[column] > 0.7)]

# Drop features 
finaldat.drop(to_drop, axis=1, inplace = True)

finaldat = pd.concat([finaldat, hold], axis = 1)
#%% In[16]:

#vis=finaldat.corr()
#vis.index
impdat = finaldat
impdat = impdat.drop(["Metro", "State", "zipcode", "City", "County"], axis = 1, )

from sklearn.preprocessing import Imputer                        #replaced NaN with mean of column
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(impdat)
train= imp.transform(impdat)
train = pd.DataFrame(train)
train.columns = impdat.columns
finaldat = pd.concat([finaldat[["Metro", "State", "zipcode", "City", "County"]], train], axis = 1)
#finaldat.iloc[:,finaldat.columns != ("Metro", "State", "zipcode", "City", "Country")] = train

finaldat.isna().sum()

#%% In[17]:

finaldat.to_csv(path_or_buf = "Desktop/final_project_data/cleaneddata.csv")
