#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 19:23:50 2020

Description:
    Re-formats PPI data for input into different PPI prediction models.
    Currently supports:
        PIPR
        SPRINT
        DEEPFE-PPI
        DPPI

@author: Eric Arezza
"""

__all__ = ['combine_pos_neg_seq', 
           'combine_pos_neg_interactions', 
           'convert_pipr', 
           'convert_sprint', 
           'convert_deepfe', 
           'convert_dppi']
__version__ = '1.0'
__author__ = 'Eric Arezza'

import os, argparse
import pandas as pd

describe_help = 'python format_interactions.py -pseq pos_sequences.fasta -nseq neg_sequences.fasta -p positive.tsv -n negative.tsv -m all'
parser = argparse.ArgumentParser(description=describe_help)
parser.add_argument('-pseq', '--pos_sequences', help='Path to positive protein sequences file (.fasta)', type=str, nargs=1)
parser.add_argument('-nseq', '--neg_sequences', help='Path to negative protein sequences file (.fasta)', type=str, nargs=1)
parser.add_argument('-p', '--positive', help='Path to positive protein interactions file (.tsv)', type=str, nargs=1)
parser.add_argument('-n', '--negative', help='Path to negative protein interactions file (.tsv)', type=str, nargs=1)
parser.add_argument('-m', '--models', help='Model for data formatting', choices=['pipr', 'sprint', 'deepfe', 'dppi', 'all'],  type=str, nargs='+')
args = parser.parse_args()

POS = args.positive[0]
NEG = args.negative[0]
POS_SEQ = args.pos_sequences[0]
NEG_SEQ = args.neg_sequences[0]
if args.models == None:
    MODELS = []
else:
    MODELS = args.models

'''
POS = 'Positives/Escherichia_coli_K12_MG1655-3.5.187_ID_511145_positive_interactions.tsv'
NEG = 'Negatives/Escherichia_coli_K12_MG1655-3.5.187_ID_511145_negative_interactions.tsv'
POS_SEQ = 'Positives/Escherichia_coli_K12_MG1655-3.5.187_ID_511145_positive_sequences.fasta'
NEG_SEQ = 'Negatives/Escherichia_coli_K12_MG1655-3.5.187_ID_511145_negative_sequences.fasta'
'''
'''
def create_cv_data(df_pos, df_neg):
    if not os.path.exists('CV_SET/'):
        os.mkdir('CV_SET/')
    np_pos = df_pos[0].to_numpy(dtype=str)
    np_neg = df_neg[0].to_numpy(dtype=str)
    kf = KFold(n_splits=5)
    fold = 1
    for train_index, test_index in kf.split(np_pos, np_neg):
        pos_train, pos_test = np_pos[train_index], np_pos[test_index]
        neg_train, neg_test = np_neg[train_index], np_neg[test_index]
        np.savetxt('CV_SET/pos_train_'+str(fold)+'.txt', pos_train, newline='\n', fmt='%s')
        np.savetxt('CV_SET/pos_test_'+str(fold)+'.txt', pos_test, newline='\n', fmt='%s')
        np.savetxt('CV_SET/neg_train_'+str(fold)+'.txt', neg_train, newline='\n', fmt='%s')
        np.savetxt('CV_SET/neg_test_'+str(fold)+'.txt', neg_test, newline='\n', fmt='%s')
        fold += 1
    print("Cross-validation subsets created!")
'''
def combine_pos_neg_seq(df_pos_seq, df_neg_seq):
    df_seq = df_pos_seq.append(df_neg_seq).reset_index(drop=True)
    prot = df_seq.iloc[::2, :].reset_index(drop=True)
    seq = df_seq.iloc[1::2, :].reset_index(drop=True)
    df = pd.DataFrame(prot[0].str.strip('>'), dtype=str)
    df.insert(1, 'Sequence', seq)
    df.drop_duplicates(inplace=True)
    return df

def combine_pos_neg_interactions(df_pos, df_neg):
    pos = df_pos.copy()
    neg = df_neg.copy()
    pos.insert(2, 'label', '1')
    pos.rename(columns={0: 'v1', 1: 'v2'}, inplace=True)
    neg.insert(2, 'label', '0')
    neg.rename(columns={0: 'v1', 1: 'v2'}, inplace=True)
    df_interactions = pos.append(neg).reset_index(drop=True)
    return df_interactions

def convert_pipr(df_pos, df_neg, df_pos_seq, df_neg_seq):
    print("Formatting datasets for PIPR...")
    formatted = 'PIPR_DATA/'
    if not os.path.exists(formatted):
        os.mkdir(formatted)
    
    df_interactions = combine_pos_neg_interactions(df_pos, df_neg)
    int_path = formatted + POS.split('/')[-1].replace('positive', 'PIPR')
    df_interactions.to_csv(int_path, sep='\t', index=False)

    df = combine_pos_neg_seq(df_pos_seq, df_neg_seq)
    seq_path = formatted + POS_SEQ.split('/')[-1].replace('positive', 'PIPR')
    df.to_csv(seq_path, sep='\t', index=False, header=False)
    print("Done!")

def convert_sprint(df_pos, df_neg, df_pos_seq, df_neg_seq):
    print("Formatting datasets for SPRINT...")
    formatted = 'SPRINT_DATA/'
    if not os.path.exists(formatted):
        os.mkdir(formatted)
    
    # Simply renames them
    pos_path = formatted + POS.split('/')[-1]
    neg_path = formatted + NEG.split('/')[-1]
    df_pos.to_csv(pos_path, sep='\t', index=False, header=False)
    df_neg.to_csv(neg_path, sep='\t', index=False, header=False)
    
    # Combine sequences and write to file
    df = combine_pos_neg_seq(df_pos_seq, df_neg_seq)
    seq_path = formatted + POS_SEQ.split('/')[-1].replace('positive', 'SPRINT')
    df.to_csv(seq_path, sep='\t', index=False, header=False)
    print("Done!")
    
def convert_deepfe(df_pos, df_neg, df_pos_seq, df_neg_seq):
    print("Formatting datasets for DEEPFE-PPI...")
    formatted = 'DEEPFE_DATA/'
    if not os.path.exists(formatted):
        os.mkdir(formatted)
    
    # Positive interactions
    protApos = pd.DataFrame(df_pos[0])
    protBpos = pd.DataFrame(df_pos[1])
    # Negative interactions
    protAneg = pd.DataFrame(df_neg[0])
    protBneg = pd.DataFrame(df_neg[1])
    
    # Combine proteins and sequences for mapping
    df = combine_pos_neg_seq(df_pos_seq, df_neg_seq)
    
    # Match proteins with sequences
    refdictseq = pd.Series(df['Sequence'].values, index=df[0]).to_dict()
    protApos.insert(1, 'Sequence', protApos[0].map(refdictseq))
    protBpos.insert(1, 'Sequence', protBpos[1].map(refdictseq))
    protAneg.insert(1, 'Sequence', protAneg[0].map(refdictseq))
    protBneg.insert(1, 'Sequence', protBneg[1].map(refdictseq))
    protApos[0] = ('>' + protApos[0])
    protBpos[1] = ('>' + protBpos[1])
    protAneg[0] = ('>' + protAneg[0])
    protBneg[1] = ('>' + protBneg[1])
    # Write to files
    pos_path = formatted + POS_SEQ.split('/')[-1].replace('sequences', 'DEEPFE').strip('.fasta')
    neg_path = formatted + NEG_SEQ.split('/')[-1].replace('sequences', 'DEEPFE').strip('.fasta')
    protApos.to_csv(pos_path + '_ProteinA.fasta', sep='\n', index=False, header=False)
    protBpos.to_csv(pos_path + '_ProteinB.fasta', sep='\n', index=False, header=False)
    protAneg.to_csv(neg_path + '_ProteinA.fasta', sep='\n', index=False, header=False)
    protBneg.to_csv(neg_path + '_ProteinB.fasta', sep='\n', index=False, header=False)
    print("Done!")
    
def convert_dppi(df_pos, df_neg, df_pos_seq, df_neg_seq):
    print("Formatting datasets for DPPI...")
    formatted = 'DPPI_DATA/'
    toblast = formatted + POS_SEQ.split('/')[-1].replace('.fasta', '/').replace('_positive', '')
    if not os.path.exists(formatted):
        os.mkdir(formatted)
    if not os.path.exists(toblast):
        os.mkdir(toblast)
    
    int_path = formatted + POS.split('/')[-1].replace('positive', 'DPPI')
    df_interactions = combine_pos_neg_interactions(df_pos, df_neg)
    df_interactions.to_csv(int_path.replace('tsv', 'csv'), index=False, header=False)
    proteins = pd.DataFrame(df_interactions['v1'].append(df_interactions['v2']).reset_index(drop=True).unique())
    proteins.to_csv(int_path.replace('_interactions', '').replace('tsv', 'node'), sep='\n', index=False, header=False)
    
    # Split fasta to separate proteins to be queried with BLAST+ to get PSSMs
    prot_pos = df_pos_seq[0][::2].reset_index(drop=True)
    seq_pos = df_pos_seq[0][1::2].reset_index(drop=True)
    prot_neg = df_neg_seq[0][::2].reset_index(drop=True)
    seq_neg = df_neg_seq[0][1::2].reset_index(drop=True)
    prot_seq_pos = pd.DataFrame(prot_pos, columns=None, dtype=str)
    prot_seq_pos.insert(1, 1, seq_pos)
    prot_seq_neg = pd.DataFrame(prot_neg, columns=None, dtype=str)
    prot_seq_neg.insert(1, 1, seq_neg)
    prot_seq = prot_seq_pos.append(prot_seq_neg)
    prot_seq.drop_duplicates(inplace=True)
    prot_seq.reset_index(drop=True, inplace=True)
    for p in range(0, prot_seq.shape[0]):
        f = open(toblast + str(prot_seq[0][p].replace('>', '')) + '.txt', 'w')
        f.write(prot_seq[0][p] + '\n' + prot_seq[1][p])
        f.close()
    print('Done!')
    

if __name__ == "__main__":
    if POS == None or NEG == None or POS_SEQ == None or NEG_SEQ == None:
        print("Missing data...run create_biogrid_dataset.py and/or provide correct file paths...")
        exit()
    if len(MODELS) == 0:
        print("Enter --model parameters to format data...")
        exit()
    
    # Read interactions.tsv and sequences.fasta to be re-formatted
    print("Reading files...")
    df_pos = pd.read_csv(POS, sep='\t', dtype=str, header=None)
    df_neg = pd.read_csv(NEG, sep='\t', dtype=str, header=None)
    df_pos_seq = pd.read_csv(POS_SEQ, sep='\t', dtype=str, header=None)
    df_neg_seq = pd.read_csv(NEG_SEQ, sep='\t', dtype=str, header=None)
        
    # Format data as per model input
    for m in MODELS:
        m = m.lower()
        if m == 'all':
            convert_pipr(df_pos, df_neg, df_pos_seq, df_neg_seq)
            convert_sprint(df_pos, df_neg, df_pos_seq, df_neg_seq)
            convert_deepfe(df_pos, df_neg, df_pos_seq, df_neg_seq)
            convert_dppi(df_pos, df_neg, df_pos_seq, df_neg_seq)
        elif m == 'pipr':
            convert_pipr(df_pos, df_neg, df_pos_seq, df_neg_seq)
        elif m == 'sprint':
            convert_sprint(df_pos, df_neg, df_pos_seq, df_neg_seq)
        elif m == 'deepfe':
            convert_deepfe(df_pos, df_neg, df_pos_seq, df_neg_seq)
        elif m == 'dppi':
            convert_dppi(df_pos, df_neg, df_pos_seq, df_neg_seq)
        else:
            print('\n', m, 'model data formatting is not available\n')
    print("Complete!")