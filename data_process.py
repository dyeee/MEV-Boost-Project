# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 01:51:34 2024

@author: S.K.
"""
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

### Data Cleaning and Transformation ###

def cleaning(bids_df, payload_df):
    # Drop null
    bids_df.dropna(subset=['value', 'num_tx', 'timestamp'], inplace=True)
    payload_df.dropna(subset=['value', 'num_tx', 'bid_timestamp_ms'], inplace=True)
    
    # Drop columns
    bids_df.drop(columns=['optimistic_submission', 'relay'], axis=1, inplace=True)
    payload_df.drop(columns=['block_datetime', 'builder_label', 'builder_pubkey', 'builder_pubkey', 'relay',
             'slot_time_ms', 'gas_limit', 'proposer_pubkey', 'proposer_mev_recipient', 'extra_data', 
             'payload_delivered', 'optimistic_submission'], axis=1, inplace=True)
    
    bids_df = bids_df.sort_values(by='block_number')
    
    print("Data cleaning has completed")    
    return bids_df, payload_df


### Data Transformation and get Derived Variables ###

def transformation(bids_df, payload_df):
    # Ensure the data types of key columns in the DataFrames are consistent
    bids_df['value'] = bids_df['value'].astype('float')
    bids_df['block_timestamp'] = pd.to_datetime(bids_df['block_timestamp'], format='mixed')
    bids_df['timestamp'] = pd.to_datetime(bids_df['timestamp'], format='mixed')
    payload_df['value'] = payload_df['value'].astype(float)
    payload_df['block_timestamp'] = pd.to_datetime(payload_df['block_timestamp'])
    payload_df['bid_timestamp_ms'] = pd.to_datetime(payload_df['bid_timestamp_ms'])
    
    # Calculate time_difference
    bids_df['time_difference'] = (bids_df['block_timestamp'] - bids_df['timestamp']).dt.total_seconds()
    payload_df['time_difference'] = (payload_df['block_timestamp'] - payload_df['bid_timestamp_ms']).dt.total_seconds()
    
    # Calculate normalized time difference
    min_time_difference_per_block = bids_df.groupby('block_number')['time_difference'].min().reset_index()
    max_time_difference_per_block = bids_df.groupby('block_number')['time_difference'].max().reset_index()
    
    bids_df = pd.merge(bids_df, min_time_difference_per_block, on='block_number', suffixes=('', '_min'))
    bids_df = pd.merge(bids_df, max_time_difference_per_block, on='block_number', suffixes=('', '_max'))
    
    bids_df['time_difference'] = 12 - bids_df['time_difference'] # avoid negative
    payload_df['time_difference'] = 12 - payload_df['time_difference']
    bids_df['time_difference_max'] = 12 - bids_df['time_difference_min'] # multiple -1
    bids_df['time_difference_min'] = 12 - bids_df['time_difference_max'] # multiple -1
    
    bids_df['normalized_t_diff'] = (bids_df['time_difference'] - bids_df['time_difference_min']) / (bids_df['time_difference_max'] - bids_df['time_difference_min'])
    
    # Calculate normalized number of transaction
    min_num_tx_per_block = bids_df.groupby('block_number')['num_tx'].min().reset_index()
    max_num_tx_per_block = bids_df.groupby('block_number')['num_tx'].max().reset_index()
    
    bids_df = pd.merge(bids_df, min_num_tx_per_block, on='block_number', suffixes=('', '_min'))
    bids_df = pd.merge(bids_df, max_num_tx_per_block, on='block_number', suffixes=('', '_max'))
    
    bids_df['normalized_num_tx'] = (bids_df['num_tx'] - bids_df['num_tx_min']) / (bids_df['num_tx_max'] - bids_df['num_tx_min'])
    
    # Calculate normalized value
    min_value_per_block = bids_df.groupby('block_number')['value'].min().reset_index()
    max_value_per_block = bids_df.groupby('block_number')['value'].max().reset_index()
    
    bids_df = pd.merge(bids_df, min_value_per_block, on='block_number', suffixes=('', '_min'))
    bids_df = pd.merge(bids_df, max_value_per_block, on='block_number', suffixes=('', '_max'))
    bids_df['normalized_value'] = (bids_df['value'] - bids_df['value_min']) / (bids_df['value_max'] - bids_df['value_min'])
    
    # Gas used ratio
    gas_limit = 30000000
    bids_df['gasUsedRatio'] = bids_df['gas_used'] / gas_limit * 100
    payload_df['gasUsedRatio'] = payload_df['gas_used'] / gas_limit * 100
    
    # Bid count per block
    bids_df['bids_count'] = bids_df.groupby('block_number')['block_number'].transform('count')
    
    # Add 'base_fee_per_gas' to bids_df
    bids_df = pd.merge(bids_df, payload_df[['block_number', 'base_fee_per_gas']], on='block_number', how='left')

    
    # Calculate value rank in each auction
    bids_df['value_rank'] = bids_df.groupby('block_number')['value'].rank(ascending=False, method='dense')
    bids_df['value_rank_percent'] = (bids_df['value_rank']/bids_df['bids_count']) * 100
    
    print("Data transformation has completed")
    return bids_df, payload_df


### Find Winners in Bids ###

def get_matched_df(bids_df, payload_df):  
    # Only extract the payload data that matches the 'block_number' in bids_df
    block_number_list = bids_df['block_number'].unique().tolist()
    print("Amount of distinct block_number in bids_df:", len(block_number_list))
    
    winner_df = payload_df[payload_df['block_number'].isin(block_number_list)] 
    winner_df = winner_df.drop_duplicates() # Make sure no duplicates
    print("Amount of matched block_number in payload_df: " ,len(winner_df))
    
    # Only extract the bids data that matches the 'block_hash' in payload_df    
    winner_block_hash_list = winner_df['block_hash'].unique().tolist()
    print("Amount of distinct winner block_hash:", len(winner_block_hash_list))
    
    # Combine bids and payloads data
    matched_df = bids_df[bids_df['block_hash'].isin(winner_block_hash_list)] 
    
    print("Got matched_df (winner bids data)")
    return matched_df

### test ###
#Load the bids and payloads CSV files

#origin_bids_df = pd.read_csv("data/Eden_MEV-boost_bid_20240404.csv") # MEV-boost bids data from Eden Public Data
#origin_payload_df = pd.read_csv("data/mev_blocks_19580000_to_19589999.csv") # MEV-Boost Winning Bid Data

#bids_df, payload_df = cleaning(origin_bids_df, origin_payload_df)
#bids_df, payload_df = transformation(bids_df, payload_df)
#matched_df = get_matched_df(bids_df, payload_df)