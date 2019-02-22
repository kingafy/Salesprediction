# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 17:18:02 2018

@author: Anshuman_Mahapatra
"""

##Predict Sales of Retail store Sales

import pandas as pd
import numpy as np
pd.options.display.max_rows
pd.options.display.max_columns

pd.set_option('display.max_columns', 30)
#Read files:
train = pd.read_csv("D:\Data Science\POC\Sales prediction\Train.csv")
print(train.head())
test = pd.read_csv("D:\Data Science\POC\Sales prediction\Test.csv")

train['source']='train'
test['source']='test'
data = pd.concat([train, test],ignore_index=True)
print(train.shape, test.shape, data.shape)

#data.to_csv("D:\Data Science\POC\Sales prediction\Merged.csv")

data.apply(lambda x: sum(x.isnull()))

data.describe()

'''
Item_Visibility has a min value of zero. This makes no practical sense because when a product is being sold in a store, the visibility cannot be 0.
Outlet_Establishment_Years vary from 1985 to 2009. The values might not be apt in this form. Rather, if we can convert them to how old the particular store is, it should have a better impact on sales.
The lower ‘count’ of Item_Weight and Item_Outlet_Sales confirms the findings from the missing value check.
'''
##Checking the variables if categorical
data.apply(lambda x: len(x.unique()))

#print(data.dtypes)

#Filter categorical variables
categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=='object']

#Exclude ID cols and source:
categorical_columns = [x for x in categorical_columns if x not in ['Item_Identifier','Outlet_Identifier','source']]

#Print frequency of categories
for col in categorical_columns:
    print('\nFrequency of Categories for varible %s'%col)
    print(data[col].value_counts())
'''
Supermarket Type1    9294
Grocery Store        1805
Supermarket Type3    1559
Supermarket Type2    1546
'''
##Data Cleaning######################################3

###ITEM WEIGHT#########################333
data.Item_Weight.plot(kind='hist', color='white', edgecolor='black', figsize=(10,6), title='Histogram of Item_Weight')
    
#Determine the average weight per item:
item_avg_weight = data.pivot_table(values='Item_Weight', index='Item_Identifier')

#Get a boolean variable specifying missing Item_Weight values
miss_bool = data['Item_Weight'].isnull() 

##print(miss_bool)

#Impute data and check #missing values before and after imputation to confirm
print('Orignal #missing: %d'% sum(miss_bool))
      
##2439

pd.set_option('display.max_rows', 5000)
print(item_avg_weight)
##print(data.loc[miss_bool,'Item_Weight'])
##data.loc[miss_bool,'Item_Weight'] = data.loc[miss_bool,'Item_Identifier'].apply(lambda x: item_avg_weight[x])
'''
data.loc[miss_bool,'Item_Weight'] = data.loc[miss_bool,'Item_Identifier'].apply(lambda x: data.groupby('Item_Identifier').mean())

pd.set_option('display.max_rows', 100)
data.loc[miss_bool,'Item_Weight'] = data.groupby([miss_bool,'Item_Identifier']).mean()
# To calculate mean use imputer class

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
'''
####Data imputation
# Create a groupby object: by_Item_Identifier
by_Item_Identifier = data.groupby(['Item_Identifier'])

# Write a function that imputes median
def impute_mean(series):
    return series.fillna(series.mean())

# Impute age and assign to titanic.age
data.Item_Weight = by_Item_Identifier.Item_Weight.transform(impute_mean)

# Print the output of titanic.tail(10)

post_clean_miss_bool = data['Item_Weight'].isnull() 
##print(post_clean_miss_bool)

print('Final #missing: %d'% sum(data['Item_Weight'].isnull()))
  
data.Item_Weight.plot(kind='hist', color='white', edgecolor='black', figsize=(10,6), title='Histogram of Item_Weight')
      
##IMPUTE MISSING OUTLETS
#Import mode function:
from scipy.stats import mode

#Determing the mode for each
print(data.Outlet_Size.dtype)
print(data.Outlet_Type.dtype)
print(data['Outlet_Type'].unique())
'''
['Supermarket Type1' 'Supermarket Type2' 'Grocery Store'
 'Supermarket Type3']
'''

print(data['Outlet_Size'].unique())

'''
['Medium' nan 'High' 'Small']
'''

##Replace all Blnaks/Nan to XXX in outlet size
data['Outlet_Size'].fillna('XXX', inplace = True)
##data['Outlet_Size'].fillNaN('XXX')

print(data['Outlet_Size'])##Check the Outlet Type and Outlet size distribution

data['COUNTER'] =1       #initially, set that counter to 1.
group_data = data.groupby(['Outlet_Type','Outlet_Size'])['COUNTER'].sum() #sum function
print(group_data)

'''
Outlet_Type        Outlet_Size
Grocery Store      Small           880
Supermarket Type1  High           1553
                   Medium         1550
                   Small          3100
Supermarket Type2  Medium         1546
Supermarket Type3  Medium         1559
Name: COUNTER, dtype: int64
'''
#Get a boolean variable specifying missing Item_Weight values
miss_bool = data['Outlet_Size'] == "XXX"

##Print Missing values
print('\nOrignal #missing: %d'% sum(miss_bool))
##Split based on Outlet type find missing outlet size
filler1 = (data['Outlet_Type'] == 'Supermarket Type1') & (data['Outlet_Size'] == 'XXX')#3091
print(filler1.sum())
filler2 = (data['Outlet_Type'] == 'Supermarket Type2') & (data['Outlet_Size'] == 'XXX')#0
print(filler2.sum())
filler3 = (data['Outlet_Type'] == 'Grocery Store') & (data['Outlet_Size'] == 'XXX')#925
print(filler3.sum())
filler4 = (data['Outlet_Type'] == 'Supermarket Type3') & (data['Outlet_Size'] == 'XXX')#0
print(filler4.sum())

##data["Outlet_Size"].fillna(data.groupby("Outlet_Type")["Outlet_Size"].transform("mode"), inplace=True)
mode1 =data.loc[data['Outlet_Type'] == 'Supermarket Type1'].Outlet_Size.mode()
print(mode1)
##Small

mode2 = data.loc[data['Outlet_Type'] == 'Supermarket Type2'].Outlet_Size.mode()
print(mode2)
## Medium

mode3 = data.loc[data['Outlet_Type'] == 'Grocery Store'].Outlet_Size.mode()
print(mode3)

#Small

mode4 = data.loc[data['Outlet_Type'] == 'Supermarket Type3'].Outlet_Size.mode()
print(mode4)


#Medium


   


    
##data(data(['Outlet_Type'] == 'Supermarket Type1').Outlet_Size.fillna('Small')
'''
data.loc[data['Outlet_Type'] == 'Supermarket Type1'].Outlet_Size.replace(np.NaN,'Small')

import numpy as np
NaN = np.nan
my_query_index = data.query('Outlet_Type == "Supermarket Type1" & Outlet_Size == @NaN').index
print(my_query_index)
data.iloc[my_query_index, 8] = 'Small'
'''

outlet_size_mode = data.pivot_table(values='Outlet_Size', columns='Outlet_Type',aggfunc=(lambda x:mode(x.astype('str')).mode[0]) )
print('Mode for each Outlet_Type:')
print(outlet_size_mode)

#Get a boolean variable specifying missing Outlet_Size values
miss_bool = data['Outlet_Size'] == "XXX" 

#Impute data and check #missing values before and after imputation to confirm
print('\nOrignal #missing: %d'%sum(miss_bool))
      
      
data['COUNTER'] =1       #initially, set that counter to 1.
group_data = data.groupby(['Outlet_Type','Outlet_Size'])['COUNTER'].sum() #sum function
print(group_data)

##apply mode to the empty ones except for Grocry store as the mode is XXX
miss_bool = data['Outlet_Size'] == "XXX"      
data.loc[miss_bool,'Outlet_Size'] = data.loc[miss_bool,'Outlet_Type'].apply(lambda x: outlet_size_mode[x])
##data.to_csv("D:\Data Science\Data\story_generation\out.csv")

data['COUNTER'] =1       #initially, set that counter to 1.
group_data = data.groupby(['Outlet_Type','Outlet_Size'])['COUNTER'].sum() #sum function
print(group_data)

#data.to_csv("D:\Data Science\Data\story_generation\out.csv")
###continue here

##data.to_csv("D:\Data Science\Data\story_generation\out.csv")
###########3PARKING POINT##############

##Change values of Outletsize XXX to Small for Grocery
data.loc[data['Outlet_Size'] == "XXX", 'Outlet_Size'] = "Small"




###data.to_csv("D:\Data Science\Data\story_generation\out.csv")

###################FEATURE ENGINNERING#############################3
data.pivot_table(values='Item_Outlet_Sales',index='Outlet_Type')


#################ITEM VISIBILITY######################
# Notice that Item_Visibility has a minimum value of 0. It seems absurd that an item has 0 
## visibility. Therefore, we will modify that column.
## Here we Group by Item_Identifier, calculate mean for each group(excluding zero values), then we proceed
## to replace the zero values in each group with the group's mean.

## we have to replace 0's by na because, mean() doesnt support exclude '0' parameter 
##but it includes exclude nan parameter which is true by default

data.loc[data.Item_Visibility == 0, 'Item_Visibility'] = np.nan

#aggregate by Item_Identifier
IV_mean = data.groupby('Item_Identifier').Item_Visibility.mean()
IV_mean

IV_mean.to_csv("D:\Data Science\POC\Sales prediction\Mean_ItemVisibility.csv")

##Clean the missing values
data.Item_Visibility.fillna(0, inplace=True)

#replace 0 values
for index, row in data.iterrows():
    if(row.Item_Visibility == 0):
        data.loc[index, 'Item_Visibility'] = IV_mean[row.Item_Identifier]
        #print(combined.loc[index, 'Item_Visibility'])
        
data.Item_Visibility.describe()
## see that min value is not zero anymore

##CREATE ITEM VISIBILITY MEAN RATIO COLUMN
print(data.head())
for index, row in data.iterrows():
    data.loc[index, 'Item_Visibility_MeanRatio'] = data.loc[index,'Item_Visibility']/IV_mean[row.Item_Identifier]
        #print(combined.loc[index, 'Item_Visibility'])
print(data.head()) 

print(data['Item_Visibility_MeanRatio'].describe())     

###CREATE BROAD CATEGORY OF TYPE OF ITEMS
##Primarily three groups FD for FOOD,DR for Drinks and NC for Non consumables


#Get the first two characters of ID:
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
#Rename them to more intuitive categories:
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
data['Item_Type_Combined'].value_counts()


##CHanging years to something reasonable. Convertng it to years of operation
#Years chosen as 2013 as the statement speaks abt 2013 collectred data
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
data['Outlet_Years'].describe()

##Data Fat
print(data['Item_Fat_Content'].unique())

##['Low Fat' 'Regular' 'low fat' 'LF' 'reg']

print('Original Categories:')
print(data['Item_Fat_Content'].value_counts())

##Converting Low fat to one generic labeling
print('\nModified Categories:')
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat','reg':'Regular','low fat':'Low Fat'})                                                      
                                                             
print(data['Item_Fat_Content'].value_counts())
'''
Modified Categories:
Low Fat    9185
Regular    5019
'''
#Mark non-consumables as separate category in low_fat:
data.loc[data['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"
data['Item_Fat_Content'].value_counts()

'''
Low Fat       6499
Regular       5019
Non-Edible    2686
'''
##Outlet Identifier
print(data.Outlet_Identifier.unique())

'''
['OUT049' 'OUT018' 'OUT010' 'OUT013' 'OUT027' 'OUT045' 'OUT017' 'OUT046'
 'OUT035' 'OUT019']
'''
##One hot coding  needed for Sklearn

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#New variable for outlet
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
le = LabelEncoder()
for i in var_mod:
    data[i] = le.fit_transform(data[i])


#One Hot Coding:
print(data.dtypes)
data = pd.get_dummies(data, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type_Combined','Outlet'])

data.dtypes

##As we have created two new columns Item_Type and Outlet Establishment year can be dropped
#Drop the columns which have been converted to different types:
data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)

#Divide into test and train:
train = data.loc[data['source']=="train"]
test = data.loc[data['source']=="test"]

#Drop unnecessary columns:
test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)


#Export files as modified versions:
train.to_csv("D:/Data Science/POC/Sales prediction/train_modified.csv",index=False)
test.to_csv("D:/Data Science/POC/Sales prediction/test_modified.csv", index=False)

print(train.head())
print(test.head())

mean_sales = train['Item_Outlet_Sales'].mean()

#Define a dataframe with IDs for submission:
base1 = test[['Item_Identifier','Outlet_Identifier']]
base1['Item_Outlet_Sales'] = mean_sales

#Export submission file
base1.to_csv("D:/Data Science/POC/Sales prediction/alg0.csv",index=False)


###Generic pipeline code for All models

target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier','Outlet_Identifier']
from sklearn import cross_validation, metrics
def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])

    #Perform cross-validation:
    cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain[target], cv=20, scoring='mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    
    #Print model report:
    print("\nModel Report")
    print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions)))
    print("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    
    #Predict on testing data:
    dtest[target] = alg.predict(dtest[predictors])
    
    #Export submission file:
    IDcol.append(target)
    submission = pd.DataFrame({ x: dtest[x] for x in IDcol})
    submission.to_csv(filename, index=False)
    


##Trying various regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
predictors = [x for x in train.columns if x not in [target]+IDcol]
# print predictors
alg1 = LinearRegression(normalize=True)
modelfit(alg1, train, test, predictors, target, IDcol, 'alg1.csv')
coef1 = pd.Series(alg1.coef_, predictors).sort_values()
coef1.plot(kind='bar', title='Model Coefficients')


##Rdge regression
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg2 = Ridge(alpha=0.05,normalize=True)
modelfit(alg2, train, test, predictors, target, IDcol, 'alg2.csv')
coef2 = pd.Series(alg2.coef_, predictors).sort_values()
coef2.plot(kind='bar', title='Model Coefficients')

from sklearn.tree import DecisionTreeRegressor
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg3 = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
modelfit(alg3, train, test, predictors, target, IDcol, 'alg3.csv')
coef3 = pd.Series(alg3.feature_importances_, predictors).sort_values(ascending=False)
coef3.plot(kind='bar', title='Feature Importances')

##Decision tree with imp variables
predictors = ['Item_MRP','Outlet_Type_0','Outlet_5','Outlet_Years']
alg4 = DecisionTreeRegressor(max_depth=8, min_samples_leaf=150)
modelfit(alg4, train, test, predictors, target, IDcol, 'alg4.csv')
coef4 = pd.Series(alg4.feature_importances_, predictors).sort_values(ascending=False)
coef4.plot(kind='bar', title='Feature Importances')

##Random forest 

from sklearn.ensemble import RandomForestRegressor
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg5 = RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4)
modelfit(alg5, train, test, predictors, target, IDcol, 'alg5.csv')
coef5 = pd.Series(alg5.feature_importances_, predictors).sort_values(ascending=False)
coef5.plot(kind='bar', title='Feature Importances')


##Random forest with other parameters
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg6 = RandomForestRegressor(n_estimators=400,max_depth=6, min_samples_leaf=100,n_jobs=4)
modelfit(alg6, train, test, predictors, target, IDcol, 'alg6.csv')
coef6 = pd.Series(alg6.feature_importances_, predictors).sort_values(ascending=False)
coef6.plot(kind='bar', title='Feature Importances')

##
##Trying Xtreme gradient boost
from sklearn import ensemble
from sklearn.utils import shuffle
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}

predictors = [x for x in train.columns if x not in [target]+IDcol]
# print predictors
alg7 = ensemble.GradientBoostingRegressor(**params)
modelfit(alg7, train, test, predictors, target, IDcol, 'alg7.csv')
coef1 = pd.Series(alg1.coef_, predictors).sort_values()
coef1.plot(kind='bar', title='Model Coefficients')


###Grid search
'''
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV
xgb1 = XGBRegressor()
params = {'nthread':[1],'min_child_weight':[4,5], 'n_estimators': [500],'gamma':[i/10.0 for i in range(3,6)],  'subsample':[i/10.0 for i in range(6,11)],'colsample_bytree':[i/10.0 for i in range(6,11)], 'max_depth': [2,4,6],'learning_rate': [.03, 0.05, .07]}
xgb_grid = GridSearchCV(xgb1,params,cv = 2,n_jobs = 5,verbose=True)
xgb_grid.fit(train[predictors], train[target])

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)

'''
'''
0.5957125782597887
{'colsample_bytree': 0.8, 'gamma': 0.3, 'learning_rate': 0.03, 'max_depth': 2, 'min_child_weight': 4, 'n_estimators': 500, 'nthread': 1, 'subsample': 1.0}
'''
from xgboost.sklearn import XGBRegressor
##from sklearn.model_selection import GridSearchCV
xgb1 = XGBRegressor()
predictors = [x for x in train.columns if x not in [target]+IDcol]
# print predictors
params = {'colsample_bytree': 0.8, 'gamma': 0.3, 'learning_rate': 0.03, 'max_depth': 2, 'min_child_weight': 4, 'n_estimators': 500, 'nthread': 1, 'subsample': 1.0}
#alg8 = xgb1(**params)
alg8 = XGBRegressor(colsample_bytree= 0.8, gamma= 0.3, learning_rate = 0.03, max_depth = 2, min_child_weight = 4, n_estimators = 500, nthread = 1, subsample = 1.0)
modelfit(alg8, train, test, predictors, target, IDcol, 'alg8.csv')
''
##coef1 = pd.Series(alg8.coef_, predictors).sort_values()
##coef1.plot(kind='bar', title='Model Coefficients')

##score 1154 for op
''



###Ensembling model for all
model1 = pd.read_csv("D:/Data Science/POC/Sales prediction/alg1.csv")
model2 = pd.read_csv("D:/Data Science/POC/Sales prediction/alg2.csv")
model3 = pd.read_csv("D:/Data Science/POC/Sales prediction/alg3.csv")
model4 = pd.read_csv("D:/Data Science/POC/Sales prediction/alg4.csv")
model5 = pd.read_csv("D:/Data Science/POC/Sales prediction/alg5.csv")
model6 = pd.read_csv("D:/Data Science/POC/Sales prediction/alg6.csv")
model7 = pd.read_csv("D:/Data Science/POC/Sales prediction/alg7.csv")
model8 = pd.read_csv("D:/Data Science/POC/Sales prediction/alg8.csv")


test_ensemble = pd.read_csv("D:/Data Science/POC/Sales prediction/alg9.csv")
##print(len(model2))
final_pred = np.array([])
for i in range(0,len(model1)):
    final_pred = (model1.Item_Outlet_Sales[i] + model2.Item_Outlet_Sales[i] + model3.Item_Outlet_Sales[i] + model4.Item_Outlet_Sales[i] + model5.Item_Outlet_Sales[i] + model6.Item_Outlet_Sales[i] + model7.Item_Outlet_Sales[i] + model8.Item_Outlet_Sales[i])/ 8
    test_ensemble.Item_Outlet_Sales[i] = final_pred
    

test_ensemble.to_csv(("D:\Data Science\POC\Sales prediction\Ensemble.csv"))


#####Trying with imp variables to Ensemble gradient Boosting

predictors = ['Item_MRP','Outlet_Type_0','Outlet_5','Outlet_Years','Outlet_Type_3','Item_Visibility','Item_Visibility_MeanRatio']

from sklearn import ensemble
from sklearn.utils import shuffle
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}

##predictors = [x for x in train.columns if x not in [target]+IDcol]
# print predictors
alg9 = ensemble.GradientBoostingRegressor(**params)
modelfit(alg9, train, test, predictors, target, IDcol, 'alg9.csv')
##coef1 = pd.Series(alg9.coef_, predictors).sort_values()
##coef1.plot(kind='bar', title='Model Coefficients')


###Ensembling model for all
##model1 = pd.read_csv("D:/Data Science/POC/Sales prediction/alg1.csv")
##model2 = pd.read_csv("D:/Data Science/POC/Sales prediction/alg2.csv")
##model3 = pd.read_csv("D:/Data Science/POC/Sales prediction/alg3.csv")
##model4 = pd.read_csv("D:/Data Science/POC/Sales prediction/alg4.csv")
##model5 = pd.read_csv("D:/Data Science/POC/Sales prediction/alg5.csv")
model6 = pd.read_csv("D:/Data Science/POC/Sales prediction/alg6.csv")
model7 = pd.read_csv("D:/Data Science/POC/Sales prediction/alg7.csv")
model8 = pd.read_csv("D:/Data Science/POC/Sales prediction/alg8.csv")
model9 = pd.read_csv("D:/Data Science/POC/Sales prediction/alg9.csv")

test_ensemble = pd.read_csv("D:/Data Science/POC/Sales prediction/alg10.csv")
##print(len(model2))
final_pred = np.array([])
for i in range(0,len(model1)):
    final_pred = (model6.Item_Outlet_Sales[i] + model7.Item_Outlet_Sales[i] + model8.Item_Outlet_Sales[i] + model9.Item_Outlet_Sales[i])/ 4
    test_ensemble.Item_Outlet_Sales[i] = final_pred
    

test_ensemble.to_csv(("D:\Data Science\POC\Sales prediction\Ensemble_3.csv"))