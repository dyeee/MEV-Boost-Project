# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 18:37:28 2024

@author: S.K.
"""
import data_process as process
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

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
    matched_df,  origin_matched_df = process.get_matched_df(bids_df, payload_df)
    
else:
    print("matched_df already exists. Skipping processing steps.")

# List of parameter sets
parameters1 = ['base_fee_per_gas', 'normalized_num_tx', 'normalized_value', 
              'gasUsedRatio','normalized_t_diff','time_difference_max',
              'bids_count']
parameters2 = ['base_fee_per_gas', 'num_tx', 'value', 
              'gasUsedRatio','time_difference','time_difference_max',
              'bids_count']

parameters = [parameters1, parameters2]

results = {}


### Feature selection ###
a = 1 # Initialised number of parameter sets
for f in parameters:
    for target in f:
        
        predictors = [p for p in f if p != target]
        
        X = matched_df[predictors]
        y = matched_df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
    
        cross_val_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        avg_cross_val_score = cross_val_scores.mean()
    
        results[target] = {
            'r2_score': r2,
            'cross_val_score': avg_cross_val_score
        }
        
        print(f"\n({a}) Results for target variable '{target}':")
        print(f"R squared error: {r2}")
        print(f"Cross-Validation Score (Negative MSE): {avg_cross_val_score}")
    a = a + 1
    
    
    print("\nSummary of results:")
    for target, scores in results.items():
        print(f"Target variable '{target}': R squared error = {scores['r2_score']}, Cross-Validation Score = {scores['cross_val_score']}")
