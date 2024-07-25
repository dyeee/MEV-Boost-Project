# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 01:58:13 2024
@author: S.K.

The purpose of this program is to verify the format of the file ethereum_mev_blocks_19580000_to_19589999.parquet, 
perform basic data processing, and save the processed data as a CSV file.

The original dataset downloaded from "MEV-Boost Winning Bid Data" on github
https://github.com/dataalways/mevboost-data/tree/main
This repository is a collection of public domain Ethereum MEV-Boost winning bid data.

"""
import os
import pandas as pd

path = 'data/'
file_name = 'ethereum__mev__blocks__19580000_to_19589999.parquet' # parquet file
csv_file = 'mev_blocks_19580000_to_19589999.csv'

dfs = [pd.read_parquet(os.path.join(path, file_name))]

# data processing
df = pd.concat(dfs)
df = df[df['payload_delivered'] == True]
df.sort_values(by=['block_number', 'bid_timestamp_ms'], ascending=True, inplace=True)
df.reset_index(inplace=True, drop=True)
df.dropna(subset='relay', inplace=True) # drop non-boost blocks
df.drop_duplicates(subset='block_hash', keep='first', inplace=True)
df.reset_index(inplace=True, drop=True) # drop relays that got the data late, only keep the earliest

print(df.info())
print(df.shape)

df.to_csv(path + csv_file, index=False)