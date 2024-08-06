# Ethereum MEV Blocks Analysis Project

This project includes several Python scripts that manage and analyze Ethereum MEV block data. Below is a brief overview of each script:

## Scripts Description

### get_parquet.py

- **Purpose**: Verifies the format of the file `ethereum_mev_blocks_19580000_to_19589999.parquet`.
- **Functionality**:
  - Performs basic data processing.
  - Saves the processed data as a CSV file.

### data_process.py

- **Purpose**: Handles data cleaning and feature engineering.
- **Functionality**:
  - Processes the payload data to identify the winning bids within the bids data.
  - Stores the results in a DataFrame called `matched_df`.

### MEV_boost_EDA.py

- **Purpose**: Performs statistical analysis on various datasets.
- **Functionality**:
  - Analyzes the `bids data`, `payload data`, and `matched_df` DataFrame.
  - Provides insights and visualizations to understand the characteristics and distributions within these datasets.

### MEV_boost_ML.py

- **Purpose**: Dedicated to model training, evaluation, and optimisation.
- **Functionality**:
  - Uses cleaned and processed data to train machine learning models.
  - Evaluates the model performances.
  - Chooses best hyperparameters and slot ranges.


----------------------------------------------------------------------------------------------
【Dataset sources】
    Eden Public Data: https://docs.edennetwork.io/public-data/overview/
    MEV-Boost Winning Bid Data: https://github.com/dataalways/mevboost-data?tab=readme-ov-file
    *This repository is a collection of public domain Ethereum MEV-Boost winning bid data.