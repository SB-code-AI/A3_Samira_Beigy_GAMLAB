import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
least_mape=1
kf= KFold(n_splits=5,shuffle=True,random_state=42)
# Load the California housing dataset
data = fetch_california_housing()
x = data.data
y = data.target
# Convert to DataFrame
df = pd.DataFrame(x, columns=data.feature_names)
#------------------------Step 1-SPLITTING_DATA------------------------
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True, random_state=42)
#---------------------------------------------------------------------
#
#
#-------------------------1-Linear_REGRESSION-------------------------
#
#
#---------------------------------------------------------------------
#------------------------Step 2-SELECTING MODEL-----------------------
from sklearn.linear_model import LinearRegression
LR_x_train=x_train
LR_y_train=y_train
LR_x_test=x_test
LR_y_test=y_test
model=LinearRegression()
#------------------------Step 3_FITTING-------------------------------
model.fit(LR_x_train, LR_y_train)
#------------------------Step 4-Prediction----------------------------
LR_y_pred = model.predict(LR_x_test)
#------------------------Step 5-Evaluation----------------------------
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
mae_test_score_LR = mean_absolute_error(LR_y_test, LR_y_pred)
print(f'MAE error for LR is: {mae_test_score_LR}')
mape_test_score_LR = mean_absolute_percentage_error(LR_y_test, LR_y_pred)
print(f'MAPE error for LR is: {mape_test_score_LR}')
if(mape_test_score_LR<least_mape):
    best_mape=mape_test_score_LR
    best_algorithm='LinearRegression'
#
#---------------------------------------------------------------------
#
#
#-------------------------2-K Neighbors-------------------------------
#
#
#---------------------------------------------------------------------
#------------------------Step 2-SELECTING MODEL-----------------------
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
knn_x_train=x_train
knn_y_train=y_train
knn_x_test=x_test
knn_y_test=y_test
#Initialize
from sklearn.model_selection import KFold
kf= KFold(n_splits=5,shuffle=True,random_state=42)
#Initialize the model
knn_model = KNeighborsRegressor()
param_grid = {
    'n_neighbors': [1, 2, 3, 4, 5, 6, 10],
    'metric': ['minkowski', 'euclidean', 'manhattan']
}
#------------------------Step 3_FITTING-------------------------------
#model.fit(knn_x_train, knn_y_train)
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(
    knn_model,
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',  # Regression metric
    cv=kf,
    n_jobs=-1
)
grid_search.fit(x_train, y_train)
#Initialize GridSearchCV
#------------------------Step 4-Prediction----------------------------
#knn_y_pred = model.predict(knn_x_test)
gs=GridSearchCV(model,param_grid,cv=kf,scoring='accuracy')
#------------------------Step 5-Evaluation----------------------------
best_params = grid_search.best_params_
best_score = grid_search.best_score_
# Use the best estimator to make predictions
best_knn_model = grid_search.best_estimator_
# Make predictions on the test data using the best KNN model
knn_y_pred = best_knn_model.predict(x_test)
# Evaluate the KNN model
mae_test_score_knn = mean_absolute_error(y_test, knn_y_pred)
mape_test_score_knn = mean_absolute_percentage_error(y_test, knn_y_pred)
print(f'MAE error for KNN: {mae_test_score_knn}')
print(f'MAPE error for KNN: {mape_test_score_knn}')
if(mape_test_score_knn<least_mape):
    best_mape=mape_test_score_knn
    best_algorithm='KNeighborsRegressor'
#---------------------------------------------------------------------
#
#
#-------------------------3-Desicion Tree-----------------------------????????????????????????????????????
#
#
#---------------------------------------------------------------------
dt_x_train=x_train
dt_x_test=x_test
dt_y_train=y_train
dt_y_test=y_test
from sklearn.tree import DecisionTreeRegressor
param_grid = {
    'max_depth': [3, 5, 7, 10, None],  # Different depths to test
    'min_samples_split': [0.05, 0.1, 0.2, 0.25],  # Proportion of samples for splitting
    'criterion': ['absolute_error', 'squared_error']  # Different criteria for the loss function
}
# Initialize K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Initialize the Decision Tree model
dt_model = DecisionTreeRegressor(random_state=43)
# Initialize GridSearchCV with the Decision Tree model, parameter grid, and K-Fold
grid_search = GridSearchCV(
    estimator=dt_model,
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',  # Suitable scoring for regression
    cv=kf,  # Using K-Fold cross-validation
    n_jobs=-1  # Use all available cores
)
# Fit the GridSearchCV on the training data
grid_search.fit(dt_x_train, dt_y_train)
# Get the best parameters and best score from GridSearchCV
best_params = grid_search.best_params_
best_score = -grid_search.best_score_  # Convert negative score back to positive
# Use the best estimator found by GridSearchCV to make predictions
best_dt_model = grid_search.best_estimator_
# Predict on the test data using the best Decision Tree model
dt_y_pred = best_dt_model.predict(x_test)
# Evaluate the Decision Tree model
mae_test_score_dt = mean_absolute_error(y_test, dt_y_pred)
mape_test_score_dt = mean_absolute_percentage_error(y_test, dt_y_pred)
print(f'MAE error for Decision Tree: {mae_test_score_dt}')
print(f'MAPE error for Decision Tree: {mape_test_score_dt}')
if(mape_test_score_dt<least_mape):
    best_mape=mape_test_score_dt
    best_algorithm='DecisionTreeRegressor'
#---------------------------------------------------------------------
#
#
#-------------------------4-RANDOM FOREST-----------------------------
#
#
#---------------------------------------------------------------------
#------------------------Step 2-SELECTING MODEL-----------------------
rf_x_train=x_train
rf_x_test=x_test
rf_y_train=y_train
rf_y_test=y_test
from sklearn.ensemble import RandomForestRegressor
# model=RandomForestRegressor(random_state=42,n_estimators=40,max_depth=10,min_samples_split=5,min_samples_leaf=3) 
# #------------------------Step 3_FITTING-------------------------------
# model.fit(rf_x_train, rf_y_train)
# #------------------------Step 4-Prediction----------------------------
# rf_y_pred = model.predict(rf_x_test)
# #------------------------Step 5-Evaluation----------------------------
# mae_test_score_random_forest = mean_absolute_error(rf_y_test, rf_y_pred)
# print(f'MAE error for Random Forest is: {mae_test_score_random_forest}')
# mape_train_score_random_forest=mean_absolute_percentage_error(dt_y_test,dt_y_pred)
# print(f'MAP Eerror for Random Forest is:{mape_train_score_random_forest}')
param_grid = {
    'n_estimators': [40, 60, 80],  # Number of trees in the forest
    'max_depth': [10, 20, None],   # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
    'min_samples_leaf': [1, 3, 5]   # Minimum samples required to be at a leaf node
}
# Initialize K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Initialize the Random Forest model
rf_model = RandomForestRegressor(random_state=42)
# Initialize GridSearchCV with the Random Forest model, parameter grid, and K-Fold
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',  # Suitable scoring for regression
    cv=kf,  # Using K-Fold cross-validation
    n_jobs=-1  # Use all available cores
)
# Fit the GridSearchCV on the training data
grid_search.fit(rf_x_train, rf_y_train)
# Get the best parameters and best score from GridSearchCV
best_params = grid_search.best_params_
best_score = -grid_search.best_score_  # Convert negative score back to positive
# Use the best estimator found by GridSearchCV to make predictions
best_rf_model = grid_search.best_estimator_
# Predict on the test data using the best Random Forest model
rf_y_pred = best_rf_model.predict(x_test)
# Evaluate the Random Forest model on the test set
mae_test_score_random_forest = mean_absolute_error(y_test, rf_y_pred)
mape_test_score_random_forest = mean_absolute_percentage_error(y_test, rf_y_pred)
print(f'MAE error for Random Forest: {mae_test_score_random_forest}')
print(f'MAPE error for Random Forest: {mape_test_score_random_forest}')
if(mape_test_score_random_forest<least_mape):
    best_mape=mape_test_score_random_forest
    best_algorithm='RandomForestRegressor'
#---------------------------------------------------------------------
#
#
#-----------------------------5-SVC-----------------------------------
#
#
#---------------------------------------------------------------------
#------------------------Step 2-SELECTING MODEL-----------------------
svr_x_train=x_train
svr_x_test=x_test
svr_y_train=y_train
svr_y_test=y_test
from sklearn.svm import SVR 
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Define the parameter grid for the SVR (optional for tuning)
param_grid = {
    'kernel': ['poly'],
    'degree': [2, 3, 4],  # Different degrees for polynomial kernel
    'C': [0.1, 1, 10],    # Regularization parameter
    'epsilon': [0.1, 0.2, 0.5]  # Epsilon for the loss function
}
# Initialize the SVR model
svr_model = SVR()
# Use GridSearchCV with K-Fold cross-validation
grid_search = GridSearchCV(
    estimator=svr_model,
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',  # Suitable for regression problems
    cv=kf,  # Use K-Fold cross-validation
    n_jobs=-1  # Use all available cores for faster computation
)
# Fit the GridSearchCV on the training data
grid_search.fit(svr_x_train, svr_y_train)
# Get the best parameters and best score from GridSearchCV
best_params = grid_search.best_params_
best_score = -grid_search.best_score_  # Convert negative score back to positive
# Use the best estimator found by GridSearchCV to make predictions
best_svr_model = grid_search.best_estimator_
# Predict on the test data using the best SVR model
svr_y_pred = best_svr_model.predict(svr_x_test)
# Evaluate the SVR model on the test set
mae_test_score_svc = mean_absolute_error(svr_y_test, svr_y_pred)
mape_test_score_svc = mean_absolute_percentage_error(svr_y_test, svr_y_pred)
print(f'MAE error for SVR: {mae_test_score_svc}')
print(f'MAPE error for SVR: {mape_test_score_svc}')
if(mape_test_score_svc<least_mape):
    best_mape=mape_test_score_svc
    best_algorithm='SVR'
#---------------------------------------------------------------------
#
#
#---------------------------------------------------------------------
print(f'based on our simulation best regression algorithm is {best_algorithm} by Mean Absolute Percentage Error about {best_mape} ')
