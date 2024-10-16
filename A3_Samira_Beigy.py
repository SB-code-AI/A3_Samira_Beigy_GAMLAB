import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

# Load the California housing dataset
data = fetch_california_housing()
x = data.data
y = data.target
# Convert to DataFrame
df = pd.DataFrame(x, columns=data.feature_names)
#print(df.columns)
#print(x)
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
mape_train_score_LR=mean_absolute_percentage_error(LR_y_test,LR_y_pred)
print(f'MAP Eerror for LR is:{mape_train_score_LR}')
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
knn_x_train=x_train
knn_y_train=y_train
knn_x_test=x_test
knn_y_test=y_test
model = KNeighborsRegressor(n_neighbors=5)
#------------------------Step 3_FITTING-------------------------------
model.fit(knn_x_train, knn_y_train)
#------------------------Step 4-Prediction----------------------------
knn_y_pred = model.predict(knn_x_test)
#------------------------Step 5-Evaluation----------------------------
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
mae_test_score_knn = mean_absolute_error(knn_y_test, knn_y_pred)
print(f'MAE error for KNN is: {mae_test_score_knn}')
mape_train_score_knn=mean_absolute_percentage_error(knn_y_test,knn_y_pred)
print(f'MAP Eerror for Desicion Tree is:{mape_train_score_knn}')
#
#---------------------------------------------------------------------
#
#
#-------------------------3-Desicion Tree-----------------------------
#
#
#---------------------------------------------------------------------
#------------------------Step 2-SELECTING MODEL-----------------------
dt_x_train=x_train
dt_x_test=x_test
dt_y_train=y_train
dt_y_test=y_test
from sklearn.tree import DecisionTreeRegressor
model =  DecisionTreeRegressor(max_depth=3,random_state=43)
#------------------------Step 3_FITTING-------------------------------
model.fit(dt_x_train, dt_y_train)
#------------------------Step 4-Prediction----------------------------
dt_y_pred = model.predict(dt_x_test)
#------------------------Step 5-Evaluation----------------------------
mae_test_score_desicion_tree = mean_absolute_error(dt_y_test, dt_y_pred)
print(f'MAE error for Decision Tree is: {mae_test_score_desicion_tree}')
mape_train_score_desicion_tree=mean_absolute_percentage_error(dt_y_test,dt_y_pred)
print(f'MAP Eerror for Desicion Tree is:{mape_train_score_desicion_tree}')
#---------------------------------------------------------------------
#
#
#
#
#
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#
#
#-------------------------3-RANDOM FOREST-----------------------------
#
#
#---------------------------------------------------------------------
#------------------------Step 2-SELECTING MODEL-----------------------
rf_x_train=x_train
rf_x_test=x_test
rf_y_train=y_train
rf_y_test=y_test
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(random_state=42,n_estimators=40)
#------------------------Step 3_FITTING-------------------------------
model.fit(rf_x_train, rf_y_train)
#------------------------Step 4-Prediction----------------------------
rf_y_pred = model.predict(rf_x_test)
#------------------------Step 5-Evaluation----------------------------
mae_test_score_random_forest = mean_absolute_error(rf_y_test, rf_y_pred)
print(f'MAE error for Random Forest is: {mae_test_score_random_forest}')
mape_train_score_random_forest=mean_absolute_percentage_error(dt_y_test,dt_y_pred)
print(f'MAP Eerror for Random Forest is:{mape_train_score_random_forest}')
#---------------------------------------------------------------------
#
#
#
#
#
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#
#
#
#
#
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#
#
#-----------------------------5-SVC-----------------------------------
#
#
#---------------------------------------------------------------------
#------------------------Step 2-SELECTING MODEL-----------------------
svc_x_train=x_train
svc_x_test=x_test
svc_y_train=y_train
svc_y_test=y_test
from sklearn.svm import SVR 
model=SVR()   
#kernel-->rbf
model=SVR(kernel='poly',degree=4) #kernelamo bokn poly , 2 
#------------------------Step 3_FITTING-------------------------------
model.fit(svc_x_train,svc_y_train)
#------------------------Step 4-Prediction----------------------------
svc_y_pred=model.predict(svc_x_test)
#------------------------Step 5-Evaluation----------------------------
mae_test_score_svc= mean_absolute_error(svc_y_test,svc_y_pred)
print(f'MAE error for SVC is: {mae_test_score_svc}')
mape_train_score_svc=mean_absolute_percentage_error(dt_y_test,dt_y_pred)
print(f'MAP Eerror for SVC is:{mape_train_score_svc}')

'''
#plt.figure(figsize=(10, 6))
plt.scatter(kn_y_test, kn_y_pred, alpha=0.6, color='b', edgecolors='k')
plt.plot([kn_y_test.min(), kn_y_test.max()], [kn_y_test.min(), kn_y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.grid(True)
plt.show()



#-------------------------SVR------------------------------
#------------------------Step3-MODEL-SELCTION----------------------------
from sklearn.svm import SVR
model=SVR(kernel='poly')
#------------------------Step4-TRAINING-----------------------------
model.fit(x_train,y_train)
#------------------------step5------------------------------
y_train_pred=model.predict(x_train)
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
mae_train_score=mean_absolute_error(y_train,y_train_pred)
print(f'MAE error for Desicion Tree is: {mae_train_score}')
mape_train_score=mean_absolute_percentage_error(y_train,y_train_pred)
print(f'MAP Eerror for Desicion Tree is:{mape_train_score}')

#-------------------------SVR------------------------------
#------------------------Step3-MODEL-SELCTION----------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import confusion_matrix

data=fetch_california_housing()
x=data.data
print(x)
y=data.target
print(y)

#data.feature_names()
df = pd.DataFrame(data.data, columns=data.feature_names)
x=data.data
y=data.target
# Print the column names
print(df.columns)
#KNN
from sklearn.neighbors import KNeighborsRegressor
model=KNeighborsRegressor(n_neighbors=3)
#step0----
#cleaning
#step1--->x , y
#step2--> train , test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25, shuffle=True, random_state=42)
#setp33-->model selection
#step4-->fitiiing
model.fit(x_train,y_train)
#step5-->validation 
y_pred = model.predict(x_test)
# You can then evaluate the model using regression metrics such as Mean Squared Error (MSE)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
'''