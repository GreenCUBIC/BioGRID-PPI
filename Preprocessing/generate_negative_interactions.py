#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 15:12:29 2020

Description:
    Randomly samples proteins in provided positive dataset
    to generate interactions not found in positives.

@author: Eric Arezza
"""

__all__ = ['generate_negatives',
           'generate_negatives_faster',
           'get_negative_sequences'
           ]
__version__ = '1.0'
__author__ = 'Eric Arezza'

import os, argparse
import pandas as pd
import numpy as np

INTRAPATH = 'Intraspecies_Interactions/'
INTERPATH = 'Interspecies_Interactions/'

describe_help = 'python generate_negative_interactions.py -pseq pos_sequences.fasta -p positive.tsv'
parser = argparse.ArgumentParser(description=describe_help)
parser.add_argument('-pseq', '--pos_sequences', help='Path to positive protein sequences file (.fasta)', type=str, nargs=1)
parser.add_argument('-p', '--positive', help='Path to positive protein interactions file (.tsv)', type=str, nargs=1)
args = parser.parse_args()

POS = args.positive[0]
POS_SEQ = args.pos_sequences[0]

#================ GENERATE NEGATIVES ==============
# Args: dataframe of positive protein interactions
# Return: dataframe of negative protein interactions
def generate_negatives(df_pos):
    proteins = df_pos[0].append(df_pos[1]).unique()
    pos_pairs = pd.Series(df_pos[0] + ' ' + df_pos[1], dtype=str)
    df_neg = pd.DataFrame()
    new_pair = np.random.choice(proteins, 2)
    neg_pairs = pd.Series(new_pair[0] + ' ' + new_pair[1])
    while(df_neg.shape[0] != df_pos.shape[0]):
        # Randomly sample a protein pair from list of proteins
        new_pair = np.random.choice(proteins, 2)
        # Check if protein pair exists in positive and negative interactions
        exists = pos_pairs[pos_pairs.str.contains(new_pair[0] + ' ' + new_pair[1])]
        exists.append(pos_pairs[pos_pairs.str.contains(new_pair[1] + ' ' + new_pair[0])])
        exists.append(neg_pairs[neg_pairs.str.contains(new_pair[0] + ' ' + new_pair[1])])
        exists.append(neg_pairs[neg_pairs.str.contains(new_pair[1] + ' ' + new_pair[0])])
        if exists.empty:
            neg_pairs = neg_pairs.append(pd.Series(new_pair[0] + ' ' + new_pair[1]), ignore_index=True)
            neg_pairs.drop_duplicates(inplace=True)
            df_neg = df_neg.append({0: new_pair[0], 1: new_pair[1]}, ignore_index=True)
        df_neg = df_neg.drop_duplicates()
        df_neg = df_neg.reset_index(drop=True)
    df_neg.to_csv(POS.replace('positive', 'negative'), sep='\t', index=False, header=False)
    print('Done!')
    return df_neg.reset_index(drop=True)

#================ GENERATE NEGATIVES FASTER ==============
# Args: dataframe of positive protein interactions
# Return: dataframe of negative protein interactions
def generate_negatives_faster(df_pos):
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
    
    # Save to file
    df_neg = df_neg[0].str.split(expand=True)
    df_neg.to_csv(POS.replace('positive', 'negative'), sep='\t', index=False, header=False)
    print('Done!')
    return df_neg

#=============== GET NEGATIVE SEQUENCES ============
# Args: dataframe of negative interactions and
#       dataframe of protein sequences
# Return: dataframe of protein sequences
def get_negative_sequences(df_neg, df_pos_seq):
    proteins = pd.Series(df_neg[0].append(df_neg[1]).unique())
    pos_prot = df_pos_seq.iloc[::2, :].reset_index(drop=True)
    pos_prot = pos_prot[0].str.replace('>', '')
    pos_seq = df_pos_seq.iloc[1::2, :].reset_index(drop=True)
    pos = pd.DataFrame({'ProteinID': pos_prot, 'Sequence': pos_seq[0]})
    neg_seq = pos[pos['ProteinID'].isin(proteins)]
    neg_seq = neg_seq.reset_index(drop=True)
    neg_seq['ProteinID'] = ('>' + neg_seq['ProteinID'])
    neg_seq.to_csv(POS_SEQ.replace('positive', 'negative'), sep='\n', index=False, header=False)
    return neg_seq

if __name__ == "__main__":
    if POS == None or POS_SEQ == None:
        print("Missing data...run get_biogrid_interactions.py and/or provide correct file paths...")
        exit()
    
    # Read interactions.tsv and sequences.fasta to be re-formatted
    print("Reading files...")
    df_pos = pd.read_csv(POS, sep='\t', dtype=str, header=None)
    df_pos_seq = pd.read_csv(POS_SEQ, sep='\t', dtype=str, header=None)
    print('Generating', df_pos.shape[0], 'negative interactions...')
    df_neg = generate_negatives_faster(df_pos)
    df_neg_seq = get_negative_sequences(df_neg, df_pos_seq)
    print('Complete!')