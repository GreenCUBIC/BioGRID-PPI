#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 12:07:42 2020

Description:
    Ensures all proteins in interaction files are found in sequenced proteins file.
    Note: Removes interactions containing unsequenced proteins.
    Re-balances positive/negative interactions.
    Intended for use after sequence homology reduction (e.g. CD-HIT).

@author: Eric Arezza
"""

__all__ = ['verify_data',
           'rebalance_set',
           'rebalance_set_faster'
           ]
__version__ = '1.0'
__author__ = 'Eric Arezza'

import os, argparse
import pandas as pd
import numpy as np

INTRAPATH = 'Intraspecies_Interactions/'
INTERPATH = 'Interspecies_Interactions/'

describe_help = 'python verify_balance_interactions.py -pseq pos_sequences.fasta -p positive.tsv -nseq neg_sequences.fasta -n negative.tsv'
parser = argparse.ArgumentParser(description=describe_help)
parser.add_argument('-pseq', '--pos_sequences', help='Path to positive protein sequences file (.fasta)', type=str, nargs=1)
parser.add_argument('-p', '--positive', help='Path to positive protein interactions file (.tsv)', type=str, nargs=1)
parser.add_argument('-nseq', '--neg_sequences', help='Path to negative protein sequences file (.fasta)', type=str, nargs=1)
parser.add_argument('-n', '--negative', help='Path to negative protein interactions file (.tsv)', type=str, nargs=1)
parser.add_argument('-o', '--output_suffix', help='Directory and suffix to add to new files (.tsv)', type=str, nargs=1)
args = parser.parse_args()

POS = args.positive[0]
POS_SEQ = args.pos_sequences[0]
NEG = args.negative[0]
NEG_SEQ = args.neg_sequences[0]
if args.output_suffix == None:
    SUFFIX = 'REDUCED'
else:
    SUFFIX = args.output_suffix[0]

# ====================== VERIFY DATA ======================
# Args: dataframes for positive and negative interactions
#       dataframes for positive and negative sequences
# Return: dataframes for positive and negative interactions
def verify_data(df_pos, df_neg, df_pos_seq, df_neg_seq):
    formatted = SUFFIX+'/'
    if not os.path.exists(formatted):
        os.mkdir(formatted)

    # Get proteins that have a sequence
    pos_prot = df_pos_seq.iloc[::2, :].reset_index(drop=True)
    pos_prot = pos_prot[0].str.replace('>', '')
    neg_prot = df_neg_seq.iloc[::2, :].reset_index(drop=True)
    neg_prot = neg_prot[0].str.replace('>', '')
    
    # Remove proteins in interactions that do not have sequence
    df_pos = df_pos[df_pos.isin(pos_prot.to_list())].dropna().reset_index(drop=True)
    df_neg = df_neg[df_neg.isin(neg_prot.to_list())].dropna().reset_index(drop=True)
    
    # Re-balance positive and negative interactions
    df_pos, df_neg = rebalance_set_faster(df_pos, df_neg)
    
    # Write out to files
    pos_path = formatted + POS.split('/')[-1].strip('.tsv') + '_'+SUFFIX+'.tsv'
    neg_path = formatted + NEG.split('/')[-1].strip('.tsv') + '_'+SUFFIX+'.tsv'
    df_pos.to_csv(pos_path, sep='\t', index=False, header=False)
    df_neg.to_csv(neg_path, sep='\t', index=False, header=False)
    print('Files added to', formatted)
    return df_pos, df_neg
    
#==================== REBALANCE SET ====================
# Args: dataframes for positive and negative datasets
# Return: dataframes for positive and negative datasets
def rebalance_set(df_pos, df_neg):
    if df_pos.shape[0] == df_neg.shape[0]:
        pass
    elif df_pos.shape[0] < df_neg.shape[0]:
        df_neg = df_neg[:df_pos.shape[0]]
    else:
        proteins = df_pos[0].append(df_neg[0].append(df_pos[1].append(df_neg[1]))).unique()
        pos_pairs = pd.Series(df_pos[0] + ' ' + df_pos[1], dtype=str)
        neg_pairs = pd.Series(df_neg[0] + ' ' + df_neg[1], dtype=str)
        print("Re-balancing data...","\nPositive interactions:", df_pos.shape[0], "\nNegative interactions:", df_neg.shape[0])
        while(df_neg.shape[0] != df_pos.shape[0]):
            # Randomly sample a protein pair from list of proteins
            new_pair = np.random.choice(proteins, 2)
            # Check if protein pair exists in interactions
            exists = pos_pairs[pos_pairs.str.contains(new_pair[0] + ' ' + new_pair[1])]
            exists.append(pos_pairs[pos_pairs.str.contains(new_pair[1] + ' ' + new_pair[0])])
            exists.append(neg_pairs[neg_pairs.str.contains(new_pair[0] + ' ' + new_pair[1])])
            exists.append(neg_pairs[neg_pairs.str.contains(new_pair[1] + ' ' + new_pair[0])])
            if exists.empty:
                df_neg = df_neg.append({0: new_pair[0], 1: new_pair[1]}, ignore_index=True)
            df_neg.drop_duplicates(inplace=True)
        print('Done!')
        print("Positive interactions:", df_pos.shape[0], "\nNegative interactions:", df_neg.shape[0])
    return df_pos, df_neg

def rebalance_set_faster(df_pos, df_neg):
    if df_pos.shape[0] == df_neg.shape[0]:
        pass
    elif df_pos.shape[0] < df_neg.shape[0]:
        df_neg = df_neg[:df_pos.shape[0]]
    else:
         # Get all proteins for sampling
        proteins = df_pos[0].append(df_pos[1]).unique()
        # Define pairs in positive interactions
        pos_pairs = pd.Series(df_pos[0] + ' ' + df_pos[1], dtype=str)
        pos_pairs_swap = pd.Series(df_pos[1] + ' ' + df_pos[0], dtype=str)
        pos = pd.DataFrame(pos_pairs.append(pos_pairs_swap, ignore_index=True), dtype=str)
        
        # Randomly sample 2 proteins to form possible new pairs
        neg = []
        for i in range(0, df_pos.shape[0]):
            pair = np.random.choice(proteins, 2)
            neg.append(pair[0] + ' ' + pair[1])
            # Add swapped for checking in positives
            neg.append(pair[1] + ' ' + pair[0])
        neg = pd.DataFrame(neg, dtype=str)
        
        # Remove pairs found in positives
        exists = neg[neg[0].isin(pos[0])]
        df_neg = neg.drop(index=exists.index)
        
        # Remove added swapped pairs
        df_neg = df_neg[df_neg.index % 2 == 0]
        # Remove other redundant pairs
        df_neg = df_neg.drop_duplicates()
        df_neg = df_neg.reset_index(drop=True)
        
        # Add new pairs until positives and negatives are balanced
        print('...generating', df_pos.shape[0] - df_neg.shape[0], 'more pairs to balance...')
        while(df_neg.shape[0] != df_pos.shape[0]):
            pair = np.random.choice(proteins, 2)
            # Check if pair exists in fully redundant positives or negatives
            if neg[neg[0].str.contains(pair[0] + ' ' + pair[1])].empty and pos[pos[0].str.contains(pair[0] + ' ' + pair[1])].empty:
                # If new pair, add to negatives
                df_neg.loc[df_neg.shape[0]] = pair[0] + ' ' + pair[1]
    
    return df_pos, df_neg

if __name__ == "__main__":
    if POS == None or NEG == None or POS_SEQ == None or NEG_SEQ == None:
        print("Missing data...run create_biogrid_dataset.py and/or provide correct file paths...")
        exit()
        
    print("Reading files...")
    df_pos = pd.read_csv(POS, sep='\t', dtype=str, header=None)
    df_neg = pd.read_csv(NEG, sep='\t', dtype=str, header=None)
    df_pos_seq = pd.read_csv(POS_SEQ, sep='\t', dtype=str, header=None)
    df_neg_seq = pd.read_csv(NEG_SEQ, sep='\t', dtype=str, header=None)
    
    print("Verifying data...")
    df_pos, df_neg = verify_data(df_pos, df_neg, df_pos_seq, df_neg_seq)
