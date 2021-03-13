from sklearn import preprocessing, metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.feature_selection import RFE
import sklearn

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks", color_codes=True)
from scipy import stats
from mp_clean_data import *
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

def linear_regression(x_train,y_train,n_features_optimal = 12):
    '''
    Linear Regression Model
    '''
    folds = KFold(n_splits = 5, shuffle = True, random_state = 100)
    hyper_params = [{'n_features_to_select': list(range(1, 15))}]
    
    chosen_params = [{'n_features_to_select': list(range(1, 15))}]
    lm = LinearRegression()
    lm.fit(x_train, y_train)
    rfe = RFE(lm)  

    model_cv = GridSearchCV(estimator = rfe, 
                            param_grid = hyper_params, 
                            scoring= 'r2', 
                            cv = folds, 
                            verbose = 1,
                            return_train_score=True) 

    model_cv.fit(x_train, y_train)
    return model_cv
    
def random_forest(max_depth, random_state,x_train,y_train):
    '''
    Random Forest Regression
    '''
    regr = RandomForestRegressor(max_depth=max_depth, random_state=random_state)
    regr.fit(x_train, y_train)
    return regr

def xgboost(x_train,y_train):
    '''
    xgboost regression
    '''
#     data_dmatrix = xgb.DMatrix(data=x_train,label=y_train)
#     data_dmatrix_test = xgb.DMatrix(data=x_test,label=y_test)

    params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
                    'max_depth': 5, 'alpha': 10, 'subsample': 1}

    regressor = xgb.XGBRegressor(
        n_estimators=100,
        reg_lambda=5,
        reg_alpha=0,
        gamma=0,
        max_depth=3
    )
    regressor.fit(x_train, y_train)
    return regressor

def ridge(x_train,y_train):
    '''
    Ridge Regression
    '''
    model_Ridge= Ridge()
    cv = 5

    # prepare a range of alpha values to test
    alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
    normalizes= ([True,False])

    ## Building Grid Search algorithm with cross-validation and Mean Squared Error score.

    grid_search_Ridge = GridSearchCV(estimator=model_Ridge,  
                         param_grid=(dict(alpha=alphas, normalize= normalizes)),
                         scoring='r2',
                         cv=cv,
                         n_jobs=-1)

    ## Lastly, finding the best parameters.

    grid_search_Ridge.fit(x_train, y_train)
    return grid_search_Ridge
    
def lasso(x_train,y_train):
    '''
    Lasso Regression
    '''
    model_Lasso= Lasso()
    cv = 5

    # prepare a range of alpha values to test
    alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
    normalizes= ([True,False])

    ## Building Grid Search algorithm with cross-validation and Mean Squared Error score.

    grid_search_lasso = GridSearchCV(estimator=model_Lasso,  
                         param_grid=(dict(alpha=alphas, normalize= normalizes)),
                         scoring='r2',
                         cv=cv,
                         n_jobs=-1)

    ## Lastly, finding the best parameters.

    grid_search_lasso.fit(x_train, y_train)
    return grid_search_lasso

def elastic(x_train,y_train):
    '''
    Elastic Regression
    '''
    model_grid_Elastic= ElasticNet()
    cv = 5
    # prepare a range of alpha values to test
    alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
    normalizes= ([True,False])

    ## Building Grid Search algorithm with cross-validation and Mean Squared Error score.

    grid_search_elastic = GridSearchCV(estimator=model_grid_Elastic,  
                         param_grid=(dict(alpha=alphas, normalize= normalizes)),
                         scoring='r2',
                         cv=cv,
                         n_jobs=-1)

    ## Lastly, finding the best parameters.

    grid_search_elastic.fit(x_train, y_train)
    return grid_search_elastic

def dataset(filename):
    '''
    Dataset 
    '''
    assert isinstance(filename,str)
    dataset, map_neigh_groups, map_neigh = cleanData(filename)
    useless_columns = ['host_id','id','last_review']
    dataset.drop(useless_columns, axis=1, inplace=True)
    all_columns = dataset.columns.to_numpy()
    features = all_columns[all_columns != 'price']
    x = dataset[features]
    y = dataset['price']
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.2,random_state=353)
    return x_train,x_test,y_train,y_test
    
    
if __name__ == '__main__':
    data_path = "./data/AB_NYC_2019.csv"
    x_train,x_test,y_train,y_test = dataset(data_path)
    
    # ----------Linear Regression---------- #
    lr = linear_regression(x_train,y_train,n_features_optimal = 12)
    y_pred = lr.predict(x_test)
    print("Linear Regression Test r2 score:", sklearn.metrics.r2_score(y_test, y_pred))
    # ------------------------------------- #
    
    # ----------Random Forest Regression---------- #
    rf = random_forest(6,0,x_train,y_train)
    y_pred = rf.predict(x_test)
    print("Random Forest Test r2 score:", sklearn.metrics.r2_score(y_test,y_pred))
    # -------------------------------------------- #
    
    # -----------Ridge Regression----------- #
    ridge = ridge(x_train,y_train)
    y_pred = ridge.predict(x_test)
    print("Ridge Regression Test r2 score:", sklearn.metrics.r2_score(y_test,y_pred))
    # -------------------------------------- #
    
    # -----------Lasso Regression----------- #
    lasso = lasso(x_train,y_train)
    y_pred = lasso.predict(x_test)
    print("Lasso Regression Test r2 score:", sklearn.metrics.r2_score(y_test,y_pred))
    # -------------------------------------- #
    
    # -----------Elastic Regression----------- #
    elastic = elastic(x_train,y_train)
    y_pred = elastic.predict(x_test)
    print("Elastic Regression Test r2 score:", sklearn.metrics.r2_score(y_test,y_pred))
    # -------------------------------------- #
    
    
    
    
    
    





      
    

