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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


import warnings
warnings.filterwarnings('ignore')

### Load the bids and payloads CSV files ###

origin_bids_df = pd.read_csv("data/Eden_MEV-Boost_bid_20240404.csv") # MEV-Boost bids data from Eden Public Data
origin_payload_df = pd.read_csv("data/mev_blocks_19580000_to_19589999.csv") # MEV-Boost Winning Bid Data

# Check if matched_df already exists in the current namespace
if 'matched_df' not in locals():
    # Process the data if matched_df does not exist
    bids_df, payload_df = process.cleaning(origin_bids_df, origin_payload_df)
    bids_df, payload_df = process.transformation(bids_df, payload_df)
    matched_df, origin_matched_df = process.get_matched_df(bids_df, payload_df)
    
else:
    print("matched_df already exists. Skipping processing steps.")



##### Build Random Forest Regression Model for ALL Matched data #####
#==============================================================================
# parameter sets
parameters1 = ['base_fee_per_gas','normalised_num_tx','normalised_value','gasUsedRatio', 'bids_count', 'normalised_t_diff'] # predictors
parameters2 = ['base_fee_per_gas','normalised_num_tx','normalised_value','gasUsedRatio', 'bids_count', 'time_difference_max']

### Choose dataset based on 'slot' ###

def dataset (slot, slot_range = 600, responser = 'time_difference_max', parameters = parameters1, df = matched_df):
    slot_range = range(slot - slot_range, slot)
    X = df[df['slot'].isin(slot_range)][parameters]
    y = df[df['slot'].isin(slot_range)][responser]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
    return X_train, X_test, y_train, y_test


### Build Random Forest Regression Model ###

# RF model (for all matched_df)
def RF_all(responser, parameters, colour, n_estimators=100, test_size=0.2, random_state=42):
    
    X = matched_df[parameters]
    y = matched_df[responser] 
    
    # Partition the data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize the Random Forest Regressor
    rf_regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    # Train the model
    rf_regressor.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_regressor.predict(X_test)

    # Evaluate the model
    error_score = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"【{responser}】")
    print(f"\nR squared error (ALL matched): {error_score}")
    print(f"Mean Squared Error (ALL matched): {mse}")
    print(f"Root Mean Squared Error (ALL matched): {rmse}")
    print("Number of training data:", len(y_train))
    print("Number of testing data:", len(y_test))

    # plot
    plt.figure(figsize=(10, 8), dpi = 600)
    plt.plot(list(y_test), color='red', label='Actual ' + responser)
    plt.plot(y_pred, color=colour, label='Predicted ' + responser)
    plt.title('Actual ' + responser + ' vs Predicted ' + responser)
    plt.xlabel('Values Count')
    plt.ylabel(responser + ' (s)')
    plt.legend()

    # Save the plot
    plt.savefig(f'graphs/ALL_matched {responser} - Random Forest Regression Model.png')
    plt.show()
    
    
# RF model (select slot range)
def RF (slot, slot_range, responser, parameters, colour):
    X_train, X_test, y_train, y_test = dataset (slot, slot_range, responser, parameters)
    
    rf_regressor = RandomForestRegressor(n_estimators=30)

    rf_regressor.fit(X_train,y_train)

    y_pred = rf_regressor.predict(X_test)

    # R squared error
    error_score = r2_score(y_test, y_pred)
    print(f"\nR squared error ({responser}, slot range = {slot_range}): ", error_score)

    print("Number of training data:", len(y_train))
    print("Number of testing data:", len(y_test))

    # plot
    plt.figure(figsize=(10, 8), dpi = 600)
    plt.plot(list(y_test), color='red', label = 'Actual ' + responser)
    plt.plot(y_pred, color=colour, label='Predicted ' + responser)
    plt.title('Actual ' + responser + ' vs Predicted ' + responser)
    plt.xlabel('Values Count')
    plt.ylabel(responser + ' (s)')
    plt.legend()

    plt.savefig(f'graphs/{responser}_{slot_range} - Random Forest Regression Model.png')
    plt.show()

#==============================================================================
# Feature Selection and plot the model prediction
RF_all('time_difference_max', parameters1, 'green')
RF_all('time_difference', parameters2, 'purple')
RF_all('normalised_t_diff', parameters2, 'blue')
RF(8787590, 600, 'time_difference_max', parameters1, 'green')
RF(8787590, 600, 'time_difference', parameters2, 'purple')
RF(8787590, 600, 'normalised_t_diff', parameters2, 'blue')
#==============================================================================



##### Extract the Best Hyperparameters and Find the Best Slot Range (using GridSearchCV) #####
#==============================================================================
def hyperparameter_tuning(X_train, y_train):
    param_grid = {
        'n_estimators': [10, 20, 30, 50, 100, 200],
        'max_depth': [10, 20, 30, 50, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestRegressor(random_state=42)

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, 
                               scoring='neg_mean_squared_error', 
                               n_jobs=-1, verbose=0)

    grid_search.fit(X_train, y_train)
    
    print("Best hyperparameters found: ", grid_search.best_params_)
    
    return grid_search.best_estimator_

def RF_turning (slot, S_range, responser, parameters, colour):
    train_range = range(100, S_range, 100) # find the best slide range
    best_train_size = 0
    best_score = -np.inf
    scores = []
    
    for train_size in train_range:
        # Adjust the dataset function as per your data processing requirements
        X_train, X_test, y_train, y_test = dataset(slot, train_size, responser, parameters)
        
        best_rf = hyperparameter_tuning(X_train, y_train)
        y_pred = best_rf.predict(X_test)
        
        score = r2_score(y_test, y_pred)
        scores.append(score)
        print(f"Slot range: {train_size}, R squared error: {score}")
        if score > best_score:
            best_score = score
            best_train_size = train_size

    plt.figure(figsize=(10, 6), dpi = 600)
    plt.plot(train_range, scores, color=colour, marker='o')
    plt.title('Model Performance vs. Slot Range (' + responser + ')')
    plt.xlabel('Slot Range')
    plt.ylabel('R Squared Error')
    plt.grid(True)
    plt.ylim(-1, 1)
    plt.savefig(f'graphs/Slot_Range_Performance {responser}.png')
    plt.show()
    print(f"\nBest train size of {responser}: {best_train_size} with R squared error: {best_score}")
    
    return best_train_size, best_rf
    
#==============================================================================
# Optimised the model
best_train_size1, best_rf1 = RF_turning(8787590, 1201, 'time_difference_max', parameters1, 'green')
best_train_size2, best_rf2 = RF_turning(8787590, 1201, 'time_difference', parameters2, 'purple')
best_train_size3, best_rf3 = RF_turning(8787590, 1201, 'normalised_t_diff', parameters2, 'blue')
#==============================================================================



### Plot for the best model ###
def plotRF (name, colour):

    plt.figure(figsize=(10, 8), dpi = 600)
    plt.plot(list(y_test), color='red', label='Actual ' + name)
    plt.plot(y_pred, color=colour, label='Predicted ' + name)
    plt.title('Actual ' + name + ' vs Predicted ' + name)
    plt.xlabel('Values Count')
    plt.ylabel(name + ' (s)')
    plt.legend()

    plt.savefig(f'graphs/Optimized_{name}_RF_Model.png')
    plt.show()
    
#==============================================================================
# Plot optimised predictive model
slot = 8787200

# time_difference_max
X_train, X_test, y_train, y_test = dataset(slot, best_train_size1)
y_pred = best_rf1.predict(X_test)

plotRF('time_difference_max', 'green')

# normalised_t_diff
X_train, X_test, y_train, y_test = dataset(slot, best_train_size2, 'time_difference', parameters2)
y_pred = best_rf2.predict(X_test)

plotRF('time_difference', 'purple')

# normalised_t_diff
X_train, X_test, y_train, y_test = dataset(slot, best_train_size3, 'normalised_t_diff', parameters2)
y_pred = best_rf3.predict(X_test)

plotRF('normalised_t_diff', 'blue')
#==============================================================================