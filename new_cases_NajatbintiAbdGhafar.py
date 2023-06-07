#%%
#1. Import packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os,datetime
from tensorflow.keras import callbacks
#%%
#2. Data loading
PATH = os.getcwd()
#%%
TRAIN_PATH = os.path.join(PATH,"cases_malaysia_train.csv")
TEST_PATH = os.path.join(PATH,"cases_malaysia_test.csv")

df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)

# %%
#3. Data inspection
#(A) Check for info
print("------------TRAIN DATA INFO-------------")
df_train.info()
print("----------------TEST DATA INFO-------------------")
df_test.info()
# %%
#(B) Check for NA values
df_train.isna().sum()
#%%
df_test.isna().sum()
# %%
#4. Data cleaning
#Fill up NA values
df_test["cases_new"] = df_test["cases_new"].interpolate()
df_test.isna().sum()
#%%
#Convert 'cases_new' column data type to int64
df_train['cases_new'] = pd.to_numeric(df_train['cases_new'], errors='coerce')
df_train['cases_new'] = df_train['cases_new'].astype('Int64')
#%%
print(df_train.dtypes)
#%%
from sklearn.impute import KNNImputer

# Select the columns to impute
columns_to_impute = ["cases_new","cluster_import", "cluster_religious","cluster_community","cluster_highRisk", "cluster_education", "cluster_detentionCentre","cluster_workplace"]

# Create an instance of the KNNImputer
imputer = KNNImputer(n_neighbors=5) 

# Perform imputation using KNN
df_train[columns_to_impute] = imputer.fit_transform(df_train[columns_to_impute])
df_train.isna().sum()
# %%
#5. Feature selection
# We are selecting "cases_new" as the feature and label
df_train_new = df_train["cases_new"]
df_test_new = df_test["cases_new"]
#%%
df_train['cases_new'].isnull().sum()
# %%
#6. Data preprocessing
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
df_train_new_scaled = mms.fit_transform(np.expand_dims(df_train_new,axis=-1))
df_test_new_scaled = mms.transform(np.expand_dims(df_test_new,axis=-1))
# %%
#7. Data windowing
window_size = 30

X_train = []
y_train = []

for i in range(window_size,len(df_train_new_scaled)):
    X_train.append(df_train_new_scaled[i-window_size:i])
    y_train.append(df_train_new_scaled[i,0])

X_train = np.array(X_train)
y_train = np.array(y_train)
# %%
df_new_stacked = np.concatenate((df_train_new_scaled,df_test_new_scaled))
length_days = window_size + len(df_test_new_scaled)
data_test = df_new_stacked[-length_days:]

X_test = []
y_test = []

for i in range(window_size,len(data_test)):
    X_test.append(data_test[i-window_size:i])
    y_test.append(data_test[i])

X_test = np.array(X_test)
y_test = np.array(y_test)
# %%
#8. Model development
from tensorflow import keras
from tensorflow.keras import Sequential,Input
from tensorflow.keras.layers import LSTM,Dropout,Dense
from tensorflow.keras.utils import plot_model
#%%
model = Sequential()
model.add(Input(shape=X_train.shape[1:]))
model.add(LSTM(64,return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64,return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(1))

model.summary()
plot_model(model,show_shapes=True,show_layer_names=True)
# %%
#9. Model compilation
model.compile(optimizer="adam",loss='mse',metrics=['mape','mse'])
#%%
#10. Create a TensorBoard callback object for the usage of TensorBoard
base_log_path = r"tensorboard_logs\new_cases_NajatbintiAbdGhafar"
log_path = os.path.join(base_log_path,datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = callbacks.TensorBoard(log_path)
# %%
#10. Model training
history = model.fit(X_train,y_train,epochs=100,callbacks=[tb])
# %%
#11. Model evaluation
print(history.history.keys())
# %%
#Plot the evaluation graph
plt.figure()
plt.plot(history.history['mape'])
plt.plot(history.history['mse'])
plt.legend(['MAPE', 'MSE'])
plt.show()

# %%
#12. Model deployment
y_pred = model.predict(X_test)
#%%
# Calulate MAPE 
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100 
# Calculate MAE
mae = np.mean(np.abs(y_test - y_pred))
# Calculate MAE 
mabe = np.mean(np.abs(y_test - np.mean(y_test)))
# Calculate MAPE Error Ratio
mape_error_ratio = ((mae - mabe) / mabe) * 100

#Check if MAPE Error Ratio is less than 1%
if mape_error_ratio < 1:
    print("MAPE is less than 1 percent for the testing dataset")
else:
    print("MAPE is not less than 1 percent for the testing dataset")

# %%
#Perform inverse transform
actual_case = mms.inverse_transform(y_test)
predicted_case = mms.inverse_transform(y_pred)
# %%
#Plot actual vs predicted
plt.figure()
plt.plot(actual_case,color='red')
plt.plot(predicted_case,color='blue')
plt.xlabel("Days")
plt.ylabel("New Cases")
plt.legend(['Actual','Predicted'])
# %%