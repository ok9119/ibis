# -*- coding: utf-8 -*-

import os
import glob
from pathlib import Path
import joblib

import pandas as pd
import numpy as np

from ibis_utils import *

def process_all_genes(base_train_path, output_base_path):
    """
    Fetch sequences for all TFs in CHS and GHTS folders

    Parameters:
    base_train_path (str)
    output_base_path (str)
    """

    folders_to_process = ['CHS', 'GHTS']

    for folder in folders_to_process:
        input_folder_path = os.path.join(base_train_path, folder)
        output_folder_path = os.path.join(output_base_path, f"{folder.lower()}_train")

        if not os.path.exists(input_folder_path):
            print(f"Folder {input_folder_path} does not exist")
            continue

        Path(output_folder_path).mkdir(parents=True, exist_ok=True)

        # List subfolders
        try:
            tfs = [d for d in os.listdir(input_folder_path)
                    if os.path.isdir(os.path.join(input_folder_path, d))]

            if not tfs:
                print(f"No subfolders in {input_folder_path}")
                continue

            # Fetching TF-bound DNA sequences
            for tf in tfs:
                try:
                    tf_path = os.path.join(input_folder_path, tf, '')
                    data = import_peak_data(tf_path)
                    preprocess_input_table(data)
                    output_file = os.path.join(output_folder_path, f"{tf}.csv")
                    data.to_csv(output_file)

                    print(f"✓ {tf} processed")

                except Exception as e:
                    print(f"✗ An error occurred while processing {tf}: {e}")

        except Exception as e:
            print(f"An error occurred while reading {input_folder_path}: {e}")

def prepare_input_tables(output_path_tokens):
    """
   Merge data from separate files containing DNA sequences which are presumably bound by different TFs and tokenize the sequences.
    """
    df_train = None
    path_list = ['/datq/ibis_train/chs_train', '/home/ok/ibis_train/ghts_train']
    df_list = []

    for path in path_list:
      file_list = Path(path).glob('*.csv')
      for file in file_list:
        df = pd.read_csv(file)
        df['TF'] = file.stem
        df_list.append(df)

    df_train = pd.concat(df_list, ignore_index=True)

    if 'Unnamed: 0' in df_train.columns:
        df_train.drop(labels='Unnamed: 0', axis=1, inplace=True)

    df_train['substring'] = df_train['substring'].map(str.upper)

    # Get reverse complementmt, sort, drop one of the complementary sequences and tokenize
    df_train['tokens'] = df_train['substring'].apply(slice_and_dice)
    df_train = df_train[['tokens', 'TF']]
    df_train.reset_index(drop=True, inplace=True)

    output_file = os.path.join(output_path_tokens, f"train_tokens.csv")
    df_train.to_csv(output_file, index=False)

    print(f"DataFrame shape: {df_train.shape}")
    print("\nПервые 5 строк:")
    print(df_train.head())

# Run:
if __name__ == "__main__":
    base_train_path = './data/ibis_train'
    output_base_path = './data/ibis_train'
    output_path_tokens = './data/ibis_train'
    process_all_genes(base_train_path, output_base_path)
    prepare_input_tables(output_path_tokens)
