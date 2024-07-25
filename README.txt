# get_parquet.py:
The purpose of this script is to verify the format of the file ethereum_mev_blocks_19580000_to_19589999.parquet, 
perform basic data processing, and save the processed data as a CSV file.

# data_process.py:
This script handles data cleaning and feature engineering. It processes the payload data to identify the winning bids within the bids data, and stores the results in a DataFrame called matched_df.

# MEV_boost_EDA.py:
This script performs statistical analysis on the bids data, payload data, and the matched_df DataFrame. It provides insights and visualizations to understand the characteristics and distributions within these datasets.

# MEV_boost_ML.py:
This script is dedicated to model training and evaluation. It uses the cleaned and processed data to train machine learning models, and evaluates their performance.
