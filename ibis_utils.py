# -*- coding: utf-8 -*-

import os
import glob
from pathlib import Path

import pandas as pd
import numpy as np

from Bio.Seq import Seq
import pysam

def fetch_sequence(data_table):
    data_table['substring']= [ref.fetch(chrom, start, end)
                               for chrom, start, end in
                               zip(data_table['#CHROM'],
                                   data_table['START_NEW'],
                                   data_table['END_NEW'])]

def preprocess_input_table(data_table):
    target_seq_len = 40
    data_table.drop(data_table.tail(1).index, inplace = True)
    data_table['abs_summit_int']= pd.to_numeric(data_table['abs_summit'])
    data_table['fold_enrichment_num']= pd.to_numeric(data_table['fold_enrichment'])

    data_table['START_NEW']= data_table['abs_summit_int'] - (target_seq_len/2)
    data_table['END_NEW']= data_table['abs_summit_int'] + (target_seq_len/2)
    data_table['log2_fold']= round(np.log2(data_table['fold_enrichment_num']))

def import_peak_data(path):
    file_list = glob.glob(os.path.join(path, '*'))

    df_list = []

    for file in file_list:
        df = pd.read_csv(file, sep = '\t')
        df.drop(df.tail(1).index, inplace = True)
        df_list.append(df)
    return pd.concat(df_list, axis = 0, ignore_index=True)

def complement(substring):
    substring = Seq(substring)
    rc = "".join(substring.reverse_complement())
    return rc

def slice_and_dice(string, window_size=8, step=1):
    xSample = []
    string_len = len(string)
    index = 0
    if step == 0:
        raise ValueError('Step must be > 0')
    else:
        while (index + window_size <= string_len):
            substring = string[index:index+window_size]
            complement_string = complement(substring)
            str_list = [substring, complement_string]
            str_list = sorted(str_list)
            xSample.append(str_list[0])
            index += step
        return xSample
