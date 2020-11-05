#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 15:12:29 2020

Description:
    Randomly samples proteins in provided positive dataset
    to generate interactions not found in positives.
    Includes option to sample from proteins with no similar
    subcellular location listed by UniProt, files will have
    the added 'seg' in name for segregated proteins.

Last Updated: Nov 2 2020

@author: Eric Arezza
"""

__all__ = ['generate_negatives',
           'get_negative_sequences',
           'get_protein_info'
           ]
__version__ = '1.0'
__author__ = 'Eric Arezza'

import sys, argparse
import pandas as pd
import urllib.parse
import urllib.request
from io import StringIO
import numpy as np
import re

INTRAPATH = 'Intraspecies_Interactions/'
INTERPATH = 'Interspecies_Interactions/'

describe_help = 'python generate_negative_interactions.py -pseq pos_sequences.fasta -p positive.tsv'
parser = argparse.ArgumentParser(description=describe_help)
parser.add_argument('-pseq', '--pos_sequences', help='Path to positive protein sequences file (.fasta)', type=str, nargs=1)
parser.add_argument('-p', '--positive', help='Path to positive protein interactions file (.tsv)', type=str, nargs=1)
parser.add_argument('-d', '--diff_subcell_local', action='store_true', help='Sample from proteins in seperate subcellular localizations')
args = parser.parse_args()

POS = args.positive[0]
POS_SEQ = args.pos_sequences[0]
LOC = args.diff_subcell_local
'''
#================ GENERATE NEGATIVES ==============
# Args: dataframe of positive protein interactions
# Return: dataframe of negative protein interactions
def generate_negatives(df_pos):
    sample_proteins = df_pos[0].append(df_pos[1]).unique()
    pos_pairs = pd.Series(df_pos[0] + ' ' + df_pos[1], dtype=str)
    df_neg = pd.DataFrame()
    if LOC:
        # Get all protein info
        df_uniprot = get_protein_info(sample_proteins)
        print(df_uniprot.shape[0], 'available proteins for sampling...')
        sample_proteins = df_uniprot['Entry'].unique()
        
    neg_pairs = pd.Series()
    while(df_neg.shape[0] != df_pos.shape[0]):
        sys.stdout.write('\r...Generating negative pairs...' + str(df_neg.shape[0]) + '/' + str(df_pos.shape[0]))
        sys.stdout.flush()
        # Randomly sample a protein pair from list of proteins
        new_pair = np.random.choice(sample_proteins, 2)
        if LOC:
            # Check if pair has info
            if df_uniprot[df_uniprot['Entry'].isin(new_pair)].shape[0] != 2:
                continue
            else:
                # Add to negatives if no similar subcellular locations
                prot_A = df_uniprot[df_uniprot['Entry'] == new_pair[0]].reset_index(drop=True)
                prot_B = df_uniprot[df_uniprot['Entry'] == new_pair[1]].reset_index(drop=True)
                if len(set(prot_A['Subcellular location [CC]'][0]) & set(prot_B['Subcellular location [CC]'][0])) == 0:
                    # Check if protein pair exists in positive and negative interactions
                    exists = pos_pairs[pos_pairs.str.contains(new_pair[0] + ' ' + new_pair[1])]
                    exists.append(pos_pairs[pos_pairs.str.contains(new_pair[1] + ' ' + new_pair[0])])
                    if len(neg_pairs) > 0:
                        exists.append(neg_pairs[neg_pairs.str.contains(new_pair[0] + ' ' + new_pair[1])])
                        exists.append(neg_pairs[neg_pairs.str.contains(new_pair[1] + ' ' + new_pair[0])])
                    if exists.empty:
                        neg_pairs = neg_pairs.append(pd.Series(new_pair[0] + ' ' + new_pair[1]), ignore_index=True)
                        neg_pairs.drop_duplicates(inplace=True)
                        df_neg = df_neg.append({0: new_pair[0], 1: new_pair[1]}, ignore_index=True)
                    df_neg = df_neg.drop_duplicates()
                    df_neg = df_neg.reset_index(drop=True)
        else:
            # Check if protein pair exists in positive and negative interactions
            exists = pos_pairs[pos_pairs.str.contains(new_pair[0] + ' ' + new_pair[1])]
            exists.append(pos_pairs[pos_pairs.str.contains(new_pair[1] + ' ' + new_pair[0])])
            if len(neg_pairs) > 0:
                exists.append(neg_pairs[neg_pairs.str.contains(new_pair[0] + ' ' + new_pair[1])])
                exists.append(neg_pairs[neg_pairs.str.contains(new_pair[1] + ' ' + new_pair[0])])
            if exists.empty:
                neg_pairs = neg_pairs.append(pd.Series(new_pair[0] + ' ' + new_pair[1]), ignore_index=True)
                neg_pairs.drop_duplicates(inplace=True)
                df_neg = df_neg.append({0: new_pair[0], 1: new_pair[1]}, ignore_index=True)
            df_neg = df_neg.drop_duplicates()
            df_neg = df_neg.reset_index(drop=True)
    # Save to file
    #df_neg.to_csv(POS.replace('positive', 'negative'), sep='\t', index=False, header=False)
    print('\nDone!')
    return df_neg.reset_index(drop=True)
'''
#================ GENERATE NEGATIVES FASTER ==============
# Args: dataframe of positive protein interactions
# Return: dataframe of negative protein interactions
def generate_negatives(df_pos):
    # Get all proteins for sampling
    sample_proteins = df_pos[0].append(df_pos[1]).unique()
    # Define pairs in positive interactions
    pos_pairs = pd.Series(df_pos[0] + ' ' + df_pos[1], dtype=str)
    pos_pairs_swap = pd.Series(df_pos[1] + ' ' + df_pos[0], dtype=str)
    pos = pd.DataFrame(pos_pairs.append(pos_pairs_swap, ignore_index=True), dtype=str)
    if LOC:
        # Get all protein info
        df_uniprot = get_protein_info(sample_proteins)
        print(df_uniprot.shape[0], 'available proteins for sampling...')
        sample_proteins = df_uniprot['Entry'].unique()
        
    df_neg = pd.DataFrame()
    while (df_neg.shape[0] < df_pos.shape[0]):
        sys.stdout.write('\r...Generating negative pairs...' + str(df_neg.shape[0]) + '/' + str(df_pos.shape[0]))
        sys.stdout.flush()
        # Randomly sample 2 proteins for possible new pairs
        neg = []
        for i in range(df_neg.shape[0], df_pos.shape[0]+1):
            pair = np.random.choice(sample_proteins, 2)
            if LOC:
                # Check if proteins found in different subcellular space, if not then get new pair
                prot_A = df_uniprot[df_uniprot['Entry'] == pair[0]].reset_index(drop=True)
                prot_B = df_uniprot[df_uniprot['Entry'] == pair[1]].reset_index(drop=True)
                if len(set(prot_A['Subcellular location [CC]'][0]) & set(prot_B['Subcellular location [CC]'][0])) != 0:
                    continue 
            neg.append(pair[0] + ' ' + pair[1])
            neg.append(pair[1] + ' ' + pair[0])
            
        neg = pd.DataFrame(neg, dtype=str)
        
        # Remove pairs found in positives
        exists = neg[neg[0].isin(pos[0])]
        df_neg_temp = neg.drop(index=exists.index)
        if df_neg.shape[0] > 0:
            # Remove pairs found in negatives
            exists = neg[neg[0].isin(df_neg[0])]
            df_neg_temp = neg.drop(index=exists.index)
        
        # Remove added swapped pairs
        df_neg_temp = df_neg_temp[df_neg_temp.index % 2 == 0]
        # Remove other redundant pairs
        df_neg_temp = df_neg_temp.drop_duplicates()
        df_neg_temp = df_neg_temp.reset_index(drop=True)
        
        # Add to negatives
        df_neg = df_neg.append(df_neg_temp).reset_index(drop=True)
    
    if df_neg.shape[0] > df_pos.shape[0]:
        df_neg = df_neg[0:df_pos.shape[0]]
    
    # Save to file
    df_neg = df_neg[0].str.split(expand=True)
    filepath = POS.replace('positive', 'negative')
    if LOC:
        filepath = POS.replace('positive', 'negative_seg')
    df_neg.to_csv(filepath, sep='\t', index=False, header=False)
    print('Done!')
    
    print('\nDone!')
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
    filepath = POS_SEQ.replace('positive', 'negative')
    if LOC:
        filepath = POS_SEQ.replace('positive', 'negative_seg')
    neg_seq.to_csv(filepath, sep='\n', index=False, header=False)
    return neg_seq

def get_protein_info(proteins):
    # Retrieve data in n proteins batches
    df = pd.DataFrame()
    if len(proteins) < 400:
        n = len(proteins)
    else:
        n = 400
    proteins_query = '+or+'.join(proteins[df.shape[0]:n])
    
    while df.shape[0] < len(proteins)-1:
        
        url = 'https://www.uniprot.org/uniprot/?query=' + proteins_query
        params = {
        'format': 'tab',
        'columns': 'id,sequence,comment(SUBCELLULAR LOCATION)'
        }
        # Request UniProt info for given organism
        sys.stdout.write('\r...Collecting protein info from UniProt...' + str(df.shape[0]) + '/' + str(len(proteins)))
        data = urllib.parse.urlencode(params)
        data = data.encode('utf-8')
        req = urllib.request.Request(url, data)
        with urllib.request.urlopen(req) as webpage:
            response = webpage.read().decode('utf-8')
        if response == '':
            print('\nNo UniProt response.')
            return pd.DataFrame()
        df_uniprot = pd.read_csv(StringIO(response), sep='\t', dtype=str)
        #df_uniprot = df_uniprot[df_uniprot['Entry'].isin(proteins[df.shape[0]:n])]
        df_uniprot = df_uniprot[df_uniprot['Entry'].isin(proteins)]
        df_uniprot.dropna(inplace=True)
        df_uniprot.reset_index(drop=True, inplace=True)
        df_uniprot['Subcellular location [CC]'] = df_uniprot['Subcellular location [CC]'].str.replace('SUBCELLULAR LOCATION: ', '')
        for i in range(0, df_uniprot.shape[0]):
            df_uniprot['Subcellular location [CC]'][i] = re.sub(r'\{[^}]*\}', '', df_uniprot['Subcellular location [CC]'][i])
            df_uniprot['Subcellular location [CC]'][i] = re.split('[?.,:;]', df_uniprot['Subcellular location [CC]'][i].lower())[:-1]
            df_uniprot['Subcellular location [CC]'][i] = [x.strip(' ') for x in df_uniprot['Subcellular location [CC]'][i]]

        df = df.append(df_uniprot)
        df.reset_index(drop=True, inplace=True)
        sys.stdout.flush()
        if (df.shape[0] + 400) > len(proteins):
            n = len(proteins)
        else:
            n += 400
        proteins_query = '+or+'.join(proteins[df.shape[0]:n])
    df.drop_duplicates(subset=['Entry'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print('\nDone!')
    
    return df

if __name__ == "__main__":
    if POS == None or POS_SEQ == None:
        print("Missing data...run get_biogrid_interactions.py and/or provide correct file paths...")
        exit()
    
    # Read interactions.tsv and sequences.fasta to be re-formatted
    print("Reading files...")
    df_pos = pd.read_csv(POS, sep='\t', dtype=str, header=None)
    df_pos_seq = pd.read_csv(POS_SEQ, sep='\t', dtype=str, header=None)
    print('Generating', df_pos.shape[0], 'negative interactions...')
    df_neg = generate_negatives(df_pos)
    df_neg_seq = get_negative_sequences(df_neg, df_pos_seq)
    print('Complete!')
