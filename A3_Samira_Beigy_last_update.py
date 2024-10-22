import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Initialize least_mape to a high value to track the best algorithm
least_mape = float('inf')  
best_algorithm = None
best_mape = None

# Load the California housing dataset
data = fetch_california_housing()
x = data.data
y = data.target

# Convert to DataFrame (optional step)
df = pd.DataFrame(x, columns=data.feature_names)

#------------------------Step 1-SPLITTING_DATA------------------------
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True, random_state=42)

#-------------------------1-Linear Regression-------------------------
model = LinearRegression()
model.fit(x_train, y_train)
LR_y_pred = model.predict(x_test)

mae_test_score_LR = mean_absolute_error(y_test, LR_y_pred)
mape_test_score_LR = mean_absolute_percentage_error(y_test, LR_y_pred)

print(f'MAE error for Linear Regression: {mae_test_score_LR}')
print(f'MAPE error for Linear Regression: {mape_test_score_LR}')

if (mape_test_score_LR < least_mape):
    least_mape = mape_test_score_LR
    best_algorithm = 'LinearRegression'
#-------------------------2-K Neighbors-------------------------------
knn_model = KNeighborsRegressor()
param_grid_knn = {
    'n_neighbors': [1, 2, 3, 4, 5, 6, 10],
    'metric': ['minkowski', 'euclidean', 'manhattan']
}
grid_search_knn = GridSearchCV(
    knn_model,
    param_grid=param_grid_knn,
    scoring='neg_mean_absolute_error',
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1
)
grid_search_knn.fit(x_train, y_train)
best_knn_model = grid_search_knn.best_estimator_
knn_y_pred = best_knn_model.predict(x_test)

mae_test_score_knn = mean_absolute_error(y_test, knn_y_pred)
mape_test_score_knn = mean_absolute_percentage_error(y_test, knn_y_pred)

print(f'MAE error for KNN: {mae_test_score_knn}')
print(f'MAPE error for KNN: {mape_test_score_knn}')

if(mape_test_score_knn < least_mape):
    least_mape = mape_test_score_knn
    best_algorithm = 'KNeighborsRegressor'

#-------------------------3-Decision Tree-----------------------------
dt_model = DecisionTreeRegressor(random_state=42)
param_grid_dt = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [0.05, 0.1, 0.2],
    'criterion': ['absolute_error', 'squared_error']
}
grid_search_dt = GridSearchCV(
    dt_model,
    param_grid=param_grid_dt,
    scoring='neg_mean_absolute_error',
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1
)
grid_search_dt.fit(x_train, y_train)
best_dt_model = grid_search_dt.best_estimator_
dt_y_pred = best_dt_model.predict(x_test)

mae_test_score_dt = mean_absolute_error(y_test, dt_y_pred)
mape_test_score_dt = mean_absolute_percentage_error(y_test, dt_y_pred)

print(f'MAE error for Decision Tree: {mae_test_score_dt}')
print(f'MAPE error for Decision Tree: {mape_test_score_dt}')

if(mape_test_score_dt < least_mape):
    least_mape = mape_test_score_dt
    best_algorithm = 'DecisionTreeRegressor'

#-------------------------4-Random Forest-----------------------------
rf_model = RandomForestRegressor(random_state=42)
param_grid_rf = {
    'n_estimators': [40, 60, 80],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5]
}
grid_search_rf = GridSearchCV(
    rf_model,
    param_grid=param_grid_rf,
    scoring='neg_mean_absolute_error',
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1
)
grid_search_rf.fit(x_train, y_train)
best_rf_model = grid_search_rf.best_estimator_
rf_y_pred = best_rf_model.predict(x_test)

mae_test_score_rf = mean_absolute_error(y_test, rf_y_pred)
mape_test_score_rf = mean_absolute_percentage_error(y_test, rf_y_pred)

print(f'MAE error for Random Forest: {mae_test_score_rf}')
print(f'MAPE error for Random Forest: {mape_test_score_rf}')

if(mape_test_score_rf < least_mape):
    least_mape = mape_test_score_rf
    best_algorithm = 'RandomForestRegressor'
    
#-------------------------5-Support Vector Regressor-------------------
svr_model = SVR()
param_grid_svr = {
    'kernel': ['poly'],
    'degree': [2, 3, 4],
    'C': [0.1, 1, 10],
    'epsilon': [0.1, 0.2, 0.5]
}
grid_search_svr = GridSearchCV(
    svr_model,
    param_grid=param_grid_svr,
    scoring='neg_mean_absolute_error',
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1
)
grid_search_svr.fit(x_train, y_train)
best_svr_model = grid_search_svr.best_estimator_
svr_y_pred = best_svr_model.predict(x_test)

mae_test_score_svr = mean_absolute_error(y_test, svr_y_pred)
mape_test_score_svr = mean_absolute_percentage_error(y_test, svr_y_pred)

print(f'MAE error for SVR: {mae_test_score_svr}')
print(f'MAPE error for SVR: {mape_test_score_svr}')

if(mape_test_score_svr < least_mape):
    least_mape = mape_test_score_svr
    best_algorithm = 'SVR'
#---------------------------------------------------------------------
print(f'Based on our simulation, the best regression algorithm is {best_algorithm} with a MAPE of {least_mape}')
