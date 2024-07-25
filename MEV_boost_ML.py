# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 01:51:55 2024

@author: S.K.

【Dataset sources】
    Eden Public Data: https://docs.edennetwork.io/public-data/overview/
    MEV-Boost Winning Bid Data: https://github.com/dataalways/mevboost-data?tab=readme-ov-file
"""
import data_process as process
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

### Load the bids and payloads CSV files ###

origin_bids_df = pd.read_csv("data/Eden_MEV-boost_bid_20240404.csv") # MEV-boost bids data from Eden Public Data
origin_payload_df = pd.read_csv("data/mev_blocks_19580000_to_19589999.csv") # MEV-Boost Winning Bid Data

# Check if matched_df already exists in the current namespace
if 'matched_df' not in locals():
    # Process the data if matched_df does not exist
    bids_df, payload_df = process.cleaning(origin_bids_df, origin_payload_df)
    bids_df, payload_df = process.transformation(bids_df, payload_df)
    matched_df = process.get_matched_df(bids_df, payload_df)
    
else:
    print("matched_df already exists. Skipping processing steps.")


### Partition the data for training and testing ###

parameters = ['base_fee_per_gas','num_tx','value','gasUsedRatio']
X = payload_df[parameters]
y = payload_df['time_difference']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)


### Build Random Forest Regression Model for ALL Payload data ###

rf_regressor = RandomForestRegressor(n_estimators=100)

rf_regressor.fit(X_train,y_train)

y_pred = rf_regressor.predict(X_test)

# R squared error
error_score = metrics.r2_score(y_test, y_pred)
print("\n R squared error (ALL payloads) : ", error_score)

y_test_list = list(y_test)
print("Number of testing data:", len(y_test_list))

# plot
plt.plot(y_test_list, color='red', label = 'Actual time_difference')
plt.plot(y_pred, color='blue', label='Predicted time_difference')
plt.title('Actual time_difference vs Predicted time_difference')
plt.xlabel('Values Count')
plt.ylabel('Time_difference')
plt.legend()
plt.show()

plt.savefig('graphs/Time_difference_ALL_payloads - Random Forest Regression Model.png')


### Choose dataset based on 'slot' ###

def dataset (slot, train_size = 600, pred_size = 30, df = payload_df, responser = 'time_difference'):
    slot_range = range(slot - train_size, slot)
    pred_slot_range = range(slot, slot + pred_size)
    parameters = ['base_fee_per_gas','num_tx','value','gasUsedRatio']
    X = df[df['slot'].isin(slot_range)][parameters]
    y = df[df['slot'].isin(slot_range)][responser]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
    X_validation = df[df['slot'].isin(pred_slot_range)][parameters]
    y_validation = df[df['slot'].isin(pred_slot_range)][responser]
    return X_train, X_test, y_train, y_test, X_validation, y_validation


### Build Random Forest Regression Model (train_size = 600) ###

slot = 8787200 # set slot number

X_train, X_test, y_train, y_test, X_validation, y_validation = dataset (slot)

rf_regressor = RandomForestRegressor(n_estimators=50)

rf_regressor.fit(X_train,y_train)

y_pred = rf_regressor.predict(X_test)

# R squared error
error_score = metrics.r2_score(y_test, y_pred)
print("\n R squared error (train_size = 600): ", error_score)

y_test_list = list(y_test)
print("Number of testing data:", len(y_test_list))

# plot
plt.figure(figsize=(10, 8))
plt.plot(y_test_list, color='red', label = 'Actual time_difference')
plt.plot(y_pred, color='blue', label='Predicted time_difference')
plt.title('Actual time_difference vs Predicted time_difference')
plt.xlabel('Values Count')
plt.ylabel('Time_difference')
plt.legend()
plt.show()

plt.savefig('graphs/Time_difference_600 - Random Forest Regression Model.png')


### Build Random Forest Regression Model (train_size = 200) ###

slot = 8787560

X_train, X_test, y_train, y_test, X_validation, y_validation = dataset (slot, 200, 20, matched_df)

rf_regressor = RandomForestRegressor(n_estimators=30)

rf_regressor.fit(X_train,y_train)

y_pred = rf_regressor.predict(X_test)

# R squared error
error_score = metrics.r2_score(y_test, y_pred)
print("\n R squared error (train_size = 200): ", error_score)

y_test_list = list(y_test)
print("Number of testing data:", len(y_test_list))

# plot
plt.figure(figsize=(10, 8))
plt.plot(y_test_list, color='red', label = 'Actual time_difference')
plt.plot(y_pred, color='blue', label='Predicted time_difference')
plt.title('Actual time_difference vs Predicted time_difference')
plt.xlabel('Values Count')
plt.ylabel('Time_difference')
plt.legend()
plt.show()

plt.savefig('graphs/Time_difference_200 - Random Forest Regression Model.png')


### Build Random Forest Regression Model (Normalized_t_diff, train_size = 200) ###

slot = 8787560

X_train, X_test, y_train, y_test, X_validation, y_validation = dataset (slot, 200, 20, matched_df, 'normalized_t_diff')

rf_regressor = RandomForestRegressor(n_estimators=30)

rf_regressor.fit(X_train,y_train)

y_pred = rf_regressor.predict(X_test)

# R squared error
error_score = metrics.r2_score(y_test, y_pred)
print("\n R squared error (Normalized_t_diff, train_size = 200): ", error_score)

y_test_list = list(y_test)
print("Number of testing data:", len(y_test_list))

# plot
plt.plot(y_test_list, color='red', label = 'Normalized_t_diff')
plt.plot(y_pred, color='blue', label='Predicted normalized_t_diff')
plt.title('Normalized time_difference vs Predicted normalized time_difference')
plt.xlabel('Values Count')
plt.ylabel('Normalized_t_diff')
plt.legend()
plt.show()

plt.savefig('graphs/Normalized_t_diff_200 - Random Forest Regression Model.png')