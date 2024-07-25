# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 01:51:54 2024

@author: S.K.

【Dataset sources】
    Eden Public Data: https://docs.edennetwork.io/public-data/overview/
    MEV-Boost Winning Bid Data: https://github.com/dataalways/mevboost-data?tab=readme-ov-file
"""
import data_process as process
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


### EDA Visualisation ###

#
sns.distplot(bids_df['gasUsedRatio'],color='green')
plt.savefig('graphs/gasUsedRatio.png')

#
sns.distplot(bids_df['num_tx'],color='blue')
plt.savefig('graphs/num_tx.png')

#
num_tx = matched_df['num_tx']
base_fee_per_gas = matched_df['base_fee_per_gas']

plt.figure(figsize=(10, 6))
plt.scatter(num_tx, base_fee_per_gas, alpha=0.5, color='blue')
plt.title('Scatter Plot of Base Fee per Gas vs Number of Transactions')
plt.xlabel('Number of Transactions')
plt.ylabel('Base Fee per Gas')
plt.grid(True)
plt.savefig('graphs/Scatter Plot of Base Fee per Gas vs Number of Transactions.png')
plt.show()

#
sns.distplot(bids_df['normalized_num_tx'],color='cyan', label='All Bids')
sns.distplot(matched_df['normalized_num_tx'], color='orange', label='Winner Bids')

plt.title('Distribution of "Normalized Number of Transactions" in Bids and Winner Bids')
plt.savefig('graphs/Distribution of Normalized_num_tx in Bids and Winner Bids.png')
plt.show()

#
sns.distplot(bids_df['value'],color='grey', label='All Bids')
sns.distplot(matched_df['value'], color='orange', label='Winner Bids')
sns.distplot(bids_df['value_max'], color='blue', label='Max bid value per block')
plt.legend()

plt.title('Distribution of "Value" in Bids and Winner Bids')
plt.savefig('graphs/Distribution of Value in Bids and Winner Bids.png')
plt.show()

#
sns.distplot(bids_df['normalized_value'],color='grey', label='All Bids')
sns.distplot(matched_df['normalized_value'], color='orange', label='Winner Bids')

plt.title('Distribution of "Normalized Value Per Block" in Bids and Winner Bids')
plt.savefig('graphs/Distribution of Normalized_Value Per Block in Bids and Winner Bids.png')
plt.show()

#
sns.distplot(bids_df['time_difference'],color='red', label='All Bids')
sns.distplot(matched_df['time_difference'],color='orange', label = 'Winner Bids')
plt.axvline(x=12, color='blue', linestyle='--', linewidth=2, label='x = 12')
plt.legend()
plt.xlim(0, 20)

plt.title('Distribution of "Bid Timestamp" in all Bids and Winner Bids')
plt.savefig('graphs/Distribution of Bid_Timestamp in all Bids and Winner Bids.png')
plt.show()

#
time_difference = matched_df['time_difference']
block_number = matched_df['block_number']

plt.figure(figsize=(10, 6))
plt.scatter(block_number, time_difference, alpha=0.5, color='blue')
plt.title('Scatter Plot of Winner Bids Timestamp vs Block Number')
plt.xlabel('Block Number')
plt.ylabel('Time Difference')
plt.grid(True)
plt.savefig('graphs/Scatter Plot of Winner Bids Timestamp vs Block Number.png')
plt.show()

#
normalized_t_diff = matched_df['normalized_t_diff']
block_number = matched_df['block_number']

plt.figure(figsize=(10, 6))
plt.scatter(block_number, normalized_t_diff, alpha=0.5, color='blue')
plt.title('Scatter Plot of Winner Bids Normalized Time Difference vs Block Number')
plt.xlabel('Block Number')
plt.ylabel('Normalized_t_diff')
plt.grid(True)
plt.savefig('graphs/Scatter Plot of Winner Bids Normalized Time Difference vs Block Number.png')
plt.show()

#
sns.distplot(bids_df['normalized_t_diff'],color='purple', label='All Bids')
sns.distplot(matched_df['normalized_t_diff'],color='orange', label = 'Winner Blocks')
plt.legend()
plt.xlim(0, 1)

plt.title('Distribution of "Normalized Time Difference" in Bids and Winner Blocks')
plt.savefig('graphs/Distribution of Normalized_t_diff in Bids and Winner Blocks')
plt.show()

#
normalized_t_diff = matched_df['normalized_t_diff']
normalized_num_tx = matched_df['normalized_num_tx']

plt.figure(figsize=(10, 6))
plt.scatter(normalized_num_tx, normalized_t_diff, alpha=0.5, color='blue')
plt.title('Scatter Plot of Winner Bids Normalized Number of Transactions vs Normalized Time Difference')
plt.xlabel('Normalized Number of Transactions')
plt.ylabel('Normalized time difference')
plt.grid(True)
plt.savefig('graphs/Scatter Plot of Winner Bids Normalized Number of Transactions vs Normalized Time Difference.png')
plt.show()

#
normalized_t_diff = matched_df['normalized_t_diff']
normalized_value = matched_df['normalized_value']

plt.figure(figsize=(10, 6))
plt.scatter(normalized_value, normalized_t_diff, alpha=0.5, color='blue')
plt.title('Scatter Plot of Normalized Value vs Winner Bids Normalized Time Difference')
plt.xlabel('Normalized value')
plt.ylabel('Normalized time difference')
plt.grid(True)
plt.savefig('graphs/Scatter Plot of Normalized Value vs Winner Bids Normalized Time Difference.png')
plt.show()

#
sns.kdeplot(
    x=normalized_value,
    y=normalized_t_diff,
    cmap="Blues",
    fill=True,
    thresh=0,
    levels=100,
    clip=((0, 1), (0, 1))
)

sns.kdeplot(
    x=normalized_value,
    y=normalized_t_diff,
    color='black',
    levels=10,
    clip=((0, 1), (0, 1))
)

plt.title('Density Plot of Winner Bids Normalized Time Difference vs Normalized Value')
plt.xlabel('Normalized value')
plt.ylabel('Normalized time difference')
plt.grid(True)
plt.savefig('graphs/Density Plot of Winner Bids Normalized Time Difference vs Normalized Value.png')
plt.show()

#
sns.distplot(matched_df['value_max'],color='purple', label='Maximum Value of Block')
sns.distplot(matched_df['value'],color='orange', label = 'Winner Bids Value')
plt.legend()

plt.title('Distribution of "Maximum Value per Block" in Bids and Winner Bids Value')
plt.savefig('graphs/Distribution of Maximum Value per Block in Bids and Winner Bids Value.png')
plt.show()

#
value_diff = (matched_df['value_max'] - matched_df['value'])
block_number = matched_df['block_number']
normalized_t_diff = matched_df['normalized_t_diff']

plt.figure(figsize=(10, 6))

sc = plt.scatter(block_number, value_diff, c=normalized_t_diff, cmap='coolwarm', alpha=0.8)
plt.colorbar(sc, label='Normalized Time Difference') # Scatter plot with normalized_t_diff represented by color

plt.title('Scatter Plot of Winner Bids Normalized Time Difference vs Maximum Value of Block')
plt.xlabel('Block number')
plt.ylabel('Maximum Value of Block - Winner Bids Value')
plt.grid(True)
plt.savefig('graphs/Scatter Plot of Winner Bids Normalized Time Difference vs Maximum Value of Block.png')
plt.show()

#
value_rank = matched_df['value_rank']
block_number = matched_df['block_number']
normalized_t_diff = matched_df['normalized_t_diff']

plt.figure(figsize=(10, 6))

sc = plt.scatter(block_number, value_rank, c=normalized_t_diff, cmap='coolwarm', alpha=0.8)
plt.colorbar(sc, label='Normalized Time Difference')

plt.title('Scatter Plot of Winner Bids Normalized Time Difference vs Winner Bids Value Rank')
plt.xlabel('Block number')
plt.ylabel('Winner Bids Value Rank')
plt.grid(True)
plt.savefig('graphs/Scatter Plot of Winner Bids Normalized Time Difference vs Winner Bids Value Rank.png')
plt.show()

#
value_rank_percent = matched_df['value_rank_percent']
block_number = matched_df['block_number']
normalized_t_diff = matched_df['normalized_t_diff']

plt.figure(figsize=(10, 6))

sc = plt.scatter(block_number, value_rank_percent, c=normalized_t_diff, cmap='coolwarm', alpha=0.8)
plt.colorbar(sc, label='Normalized Time Difference')

plt.title('Scatter Plot of Winner Bids Normalized Time Difference vs Winner Bids Value Rank (%)')
plt.xlabel('Block number')
plt.ylabel('Winner Bids Value Rank (%)')
plt.grid(True)
plt.savefig('graphs/Scatter Plot of Winner Bids Normalized Time Difference vs Winner Bids Value Rank (percent).png')
plt.show()

#
sns.kdeplot(
    x=value_rank_percent,
    y=normalized_t_diff,
    cmap="Blues",
    fill=True,
    thresh=0,
    levels=100,
    clip=((0, 100), (0, 1))
)

sns.kdeplot(
    x=value_rank_percent,
    y=normalized_t_diff,
    color='black',
    levels=10,
    clip=((0, 100), (0, 1))
)

plt.title('Density Plot of Winner Bids Normalized Time Difference vs Winner Bids Value Rank (%)')
plt.xlabel('Winner Bids Value Rank (%)')
plt.ylabel('Winner Bids Normalized Time Difference')
plt.grid(True)
plt.savefig('graphs/Density Plot of Winner Bids Normalized Time Difference vs Winner Bids Value Rank (percent).png')
plt.show()

#
numeric_cols = [ "time_difference", "normalized_t_diff", "normalized_value", "num_tx", "normalized_num_tx", "gasUsedRatio", "value", "bids_count", "base_fee_per_gas"]
corr_matrix = bids_df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, annot=True, fmt=".2f")
plt.title('Correlation Matrix')
plt.savefig('graphs/Correlation (time_difference).png')
plt.show()

#
numeric_cols = [ "normalized_t_diff", "time_difference", "normalized_value", "num_tx", "normalized_num_tx", "gasUsedRatio", "value", "bids_count", "base_fee_per_gas"]
corr_matrix = bids_df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, annot=True, fmt=".2f")
plt.title('Correlation Matrix')
plt.savefig('graphs/Correlation (normalized_t_diff).png')
plt.show()