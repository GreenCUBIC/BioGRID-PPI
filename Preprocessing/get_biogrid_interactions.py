#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 23:56:42 2020

Description:
    Collects and filters positive protein interaction datasets 
    from BioGRID files using the UniProt database for mapping
    entrez gene IDs to protein IDs and sequences.

@author: Eric Arezza
Last Updated: Nov 7 2020
"""

__all__ = ['get_biogrid_data', 
           'separate_species_interactions', 
           'get_organism_proteome', 
           'map_bgup', 
           'verify_sequences'
           ]
__version__ = '2.0'
__author__ = 'Eric Arezza'


import os, sys
import pandas as pd
import numpy as np
import urllib.parse
import urllib.request
from io import StringIO
from itertools import combinations

# To store files
PROTEOMEPATH = 'Proteomes/'
INTRAPATH = 'Intraspecies_Interactions/'
INTERPATH = 'Interspecies_Interactions/'

# For release version column names
ORGANISM_ID_A = 'Organism Interactor A'
ORGANISM_ID_B = 'Organism Interactor B'
PUBMED = 'Publication Source'

TAB2COLS = [
    'Entrez Gene Interactor A', 
    'Entrez Gene Interactor B', 
    'Pubmed ID', 
    'Organism Interactor A', 
    'Organism Interactor B']
TAB3COLS = [
    'Entrez Gene Interactor A', 
    'Entrez Gene Interactor B', 
    'Publication Source', 
    'Organism Interactor A', 
    'Organism Interactor B']
TAB3COLS_v4 = [
    'Entrez Gene Interactor A', 
    'Entrez Gene Interactor B', 
    'Publication Source', 
    'Organism ID Interactor A', 
    'Organism ID Interactor B']
    
# Define conservative filters as proposed in POSITOME
INTERACTION_TYPES=['physical']
DETECTION_METHODS=[
    'two-hybrid',
    'affinity capture-ms',
    'affinity capture-western',
    'reconstituted complex',
    'affinity capture-luminescence',
    'co-crystal structure',
    'far western',
    'fret',
    'protein-peptide',
    'co-localization',
    'affinity capture-rna',
    'co-purification'
    ]
THROUGHPUT_LEVELS=[
    'high throughput',
    'low throughput'
    ]

#=================== GET BIOGRID INTERACTIONS ==================
# Args: path to biogrid files, a list of the file names, and
#        desired use of filters true/false.
# Return: dictionary with keys as biogrid organism name and 
#         values as dataframe of interaction data
def get_biogrid_data(path, files, filters=True):
    biogrid_dfs = {}
    global ORGANISM_ID_A
    global ORGANISM_ID_B
    global PUBMED
    # Iterate over each file
    for file in files:
        print('Reading', file)
        # Read file into dataframe
        df = pd.read_csv(path + file, sep='\t', dtype=str)
        if '.tab2' in file:
            suffix = '.tab2.txt'
        elif '.tab3' in file:
            suffix = '.tab3.txt'
            
        if filters:
            # Apply filters
            print('Applying filters...')
            df = df[df['Experimental System Type'].str.lower().isin(INTERACTION_TYPES)]
            df = df[df['Experimental System'].str.lower().isin(DETECTION_METHODS)]
            df = df[df['Throughput'].str.lower().isin(THROUGHPUT_LEVELS)]
            # Filter for interactions with more than 1 Publication Source
            if '.tab2' in file:
                df = df[TAB2COLS]
                PUBMED = 'Pubmed ID'
            elif '.tab3' in file:
                if file.split('-')[-1].split('.')[0] == '3':
                    df = df[TAB3COLS]
                    ORGANISM_ID_A = 'Organism Interactor A'
                    ORGANISM_ID_B = 'Organism Interactor B'
                elif file.split('-')[-1].split('.')[0] == '4':
                    df = df[TAB3COLS_v4]
                    ORGANISM_ID_A = 'Organism ID Interactor A'
                    ORGANISM_ID_B = 'Organism ID Interactor B'
                PUBMED = 'Publication Source'

            biogrid_dfs[file.replace('BIOGRID-ORGANISM-', '').replace(suffix, '')] = df.reset_index(drop=True)
        else:
            biogrid_dfs[file.replace('BIOGRID-ORGANISM-', '').replace(suffix, '')] = df.reset_index(drop=True)
        print('Finished reading', file)
    return biogrid_dfs

#========================== CHECK MULTIPLE SOURCES ========================
# Args: BioGRID dataframe
# Return: BioGRID dataframe with non-redundant, multi-sourced interactions
#==========================================================================
def check_multiple_sources(df):
    print('Checking interactions for multiple sources, this may take a while...')
    # Keep interactions listed more than once (reduces df)
    df = df[df.duplicated(subset=['Entrez Gene Interactor A', 'Entrez Gene Interactor B'], keep=False)]
    df.insert(0, 0, df['Entrez Gene Interactor A'] + ' ' + df['Entrez Gene Interactor B'])
    interactions = df[0].unique()
    multi = []
    # Keep interactions with more than one Publication Source
    for i in interactions:
        sys.stdout.write('\rInteraction ' + str(list(interactions).index(i) + 1) + '/' + str(len(interactions)))
        sys.stdout.flush()
        if len(df[df[0] == i][PUBMED].unique()) > 1:
            multi.append(df[df[0] == i][PUBMED].index[0])
    df = df.loc[multi].reset_index(drop=True)
    df.drop(columns=[0], inplace=True)
    # Remove redundant interactions where (A-B == B-A)
    nr = pd.DataFrame(np.sort(df[['Entrez Gene Interactor A', 'Entrez Gene Interactor B']], axis=1), columns=['Entrez Gene Interactor A', 'Entrez Gene Interactor B']).drop_duplicates()
    df = df.loc[nr.index].reset_index(drop=True)
    
    return df

#====================== SEPARATE SPECIES INTERACTIONS ==========================
# Args: key (organism name), value (dataframe) generated from get_biogrid_interactions
# Return: list of dataframes with intraspecies interactions and 
#         list of dataframes with interspecies interactions
def separate_species_interactions(organismRelease, df_biogrid):
    intra_species = []
    inter_species = []
    organisms = df_biogrid[ORGANISM_ID_A].append(df_biogrid[ORGANISM_ID_B]).unique().astype(str)
    single = False
    try:
        main_organism = df_biogrid[ORGANISM_ID_A].append(df_biogrid[ORGANISM_ID_B]).mode().astype(str)[0]
        single = True
    except:
        print('Checking interspecies...')
    for organism in organisms:
        print('Pulling interactions for organism', organism)
        intra = df_biogrid.loc[(df_biogrid[ORGANISM_ID_A] == organism) & (df_biogrid[ORGANISM_ID_B] == organism)]
        
        intra = check_multiple_sources(intra)
        # Remove redundant interactions
        intra = intra.drop_duplicates(subset=['Entrez Gene Interactor A', 'Entrez Gene Interactor B'])
        intra.reset_index(drop=True, inplace=True)
        if not intra.empty:
            # Remove redundant interactions
            nr = pd.DataFrame(np.sort(intra[['Entrez Gene Interactor A', 'Entrez Gene Interactor B']], axis=1), columns=['Entrez Gene Interactor A', 'Entrez Gene Interactor B']).drop_duplicates()
            intra = intra.loc[nr.index].reset_index(drop=True)
            intra_species.append(intra)
            
    # Get combinations of interspecies pairs with main organismRemove interactions with only 1 source
    combination_pairs = list(combinations(organisms, 2))
    organism_pairs = []
    if single:
        for pair in combination_pairs:
            if main_organism in pair:
                organism_pairs.append(pair)
    else:
        organism_pairs = combination_pairs
    if len(combination_pairs) > 0:
        for pair in organism_pairs:
            print('Pulling interactions for organism pair:', pair)
            inter = pd.DataFrame()
            # Isolate interactions for interspecies pair
            inter = df_biogrid.loc[(df_biogrid[ORGANISM_ID_A] == pair[0]) & (df_biogrid[ORGANISM_ID_B] == pair[1])]
            inter = inter.append(df_biogrid.loc[(df_biogrid[ORGANISM_ID_A] == pair[1]) & (df_biogrid[ORGANISM_ID_B] == pair[0])])
            
            inter = check_multiple_sources(inter)
            # Remove redundant interactions
            inter = inter.drop_duplicates(subset=['Entrez Gene Interactor A', 'Entrez Gene Interactor B'])
            inter.reset_index(drop=True, inplace=True)
            if not inter.empty:
                # Remove redundant interactions
                nr = pd.DataFrame(np.sort(inter[['Entrez Gene Interactor A', 'Entrez Gene Interactor B']], axis=1), columns=['Entrez Gene Interactor A', 'Entrez Gene Interactor B']).drop_duplicates()
                inter = inter.loc[nr.index].reset_index(drop=True)
                inter_species.append(inter)
    return intra_species, inter_species
'''
#============ REMOVE REDUNDANT INTERACTIONS =============
# Iteratively remove ProteinA - ProteinB interaction if
# ProteinB - ProteinA interaction exists.
def remove_redundant_interactions(df):
    proteins = (df['Entrez Gene Interactor A'], df['Entrez Gene Interactor B'])
    swapped_interactions = pd.DataFrame()
    for i in range(0, len(proteins[0])):
        swapped = df.loc[(df['Entrez Gene Interactor A'] == proteins[1][i]) & (df['Entrez Gene Interactor B'] == proteins[0][i])]
        swapped_interactions = swapped_interactions.append(swapped)
    swapped_interactions = swapped_interactions[swapped_interactions['Entrez Gene Interactor A'] != swapped_interactions['Entrez Gene Interactor B']]
    df = df.drop(index=swapped_interactions.index).reset_index(drop=True)
    
    nr = pd.DataFrame(np.sort(df[['Entrez Gene Interactor A', 'Entrez Gene Interactor B']], axis=1), columns=['Entrez Gene Interactor A', 'Entrez Gene Interactor B']).drop_duplicates()
    df = df.loc[nr.index].reset_index(drop=True)
    
    return df
'''
#=================== GET ORGANISM PROTEOME =======================
# Args: organism name or ID, and if proteins are reviewed (Swiss-Prot)
# Return: dataframe of proteins and their sequence
# Note: Writes out .fasta of sequences (non-reference proteome)
def get_organism_proteome(organismRelease, reviewed='yes'):
    filepath = PROTEOMEPATH + organismRelease + '_non-reference-proteome.fasta'
    df_uniprot = pd.DataFrame()
    if os.path.isfile(filepath):
        df_uniprot = pd.read_csv(filepath, sep='\n', header=None)
        df_uniprot = pd.DataFrame(data={'Entry': df_uniprot.iloc[::2, :][0].str.replace('>', '').reset_index(drop=True), 'Sequence': df_uniprot.iloc[1::2, :][0].reset_index(drop=True)})
        return df_uniprot

    # Setup query to UniProt
    organism = organismRelease.split('-')[0].replace('_', '+')
    url = 'https://www.uniprot.org/uniprot/?query=reviewed:'+reviewed.lower()+'+AND+'+str(organism)
    params = {
    'format': 'tab',
    'columns': 'id,length,sequence,database(GeneID)'
    }
    # Request UniProt info for given organism
    print('...Collecting UniProt info for organismID: '+ organism.replace('+', ' ') +'...')
    data = urllib.parse.urlencode(params)
    data = data.encode('utf-8')
    req = urllib.request.Request(url, data)
    with urllib.request.urlopen(req) as webpage:
        response = webpage.read().decode('utf-8')
    if response == '':
        print('No UniProt results found for organism:', organism.replace('+', ' '), 'with reviewed='+reviewed+'...')
        return pd.DataFrame()
    df_uniprot = pd.read_csv(StringIO(response), sep='\t', dtype=str)
    # Remove proteins with no sequence or missing info
    df_uniprot.dropna(subset=['Entry', 'Cross-reference (GeneID)', 'Sequence']).reset_index(drop=True, inplace=True)
    df_uniprot.drop_duplicates(subset=['Entry', 'Sequence'], inplace=True)
    df_uniprot = df_uniprot.reset_index(drop=True)
        
    # Write out protein sequences to .fasta file
    if not df_uniprot.empty:
        df_uniprot_copy = df_uniprot.copy()
        df_uniprot_copy['Entry'] = ('>' + df_uniprot_copy['Entry'])
        df_uniprot_copy.to_csv(filepath, columns=['Entry', 'Sequence'], sep='\n', header=False, index=False)
        print('Organism', organismRelease, "proteome added to", PROTEOMEPATH)

    return df_uniprot

#======================= MAP BIOGRID TO UNIPROT ===============================
# Args: organism name for write out filenames and dataframe of biogrid interactions
# Return: dataframe of interactions by protein ID and dataframe of protein sequences
# Note: Queries UniProt, looks for protein ID for matching entrez gene ID, checks
#       if protein has sequence, creates a sequences.fasta and interactions.tsv
def map_bgup(organismRelease, df_biogrid):
    # Organism IDs and path for naming files
    organisms = df_biogrid[ORGANISM_ID_A].append(df_biogrid[ORGANISM_ID_B]).unique()
    organisms_name = '-'.join(organisms)
    if len(organisms) > 1:
        path = INTERPATH
    else:
        path = INTRAPATH
    
    # Query UniProt mapping
    geneIDs = df_biogrid['Entrez Gene Interactor A'].append(df_biogrid['Entrez Gene Interactor B']).unique()
    geneIDs = str(geneIDs.tolist()).strip('[').strip(']').replace("'", "").replace(',', '')
    url = 'https://www.uniprot.org/uploadlists/'
    params = {
    'from': 'P_ENTREZGENEID',
    'to': 'ACC',
    'format': 'tab',
    'columns': 'id,sequence,reviewed',
    'query': geneIDs,
    }
    data = urllib.parse.urlencode(params)
    data = data.encode('utf-8')
    req = urllib.request.Request(url, data)
    with urllib.request.urlopen(req) as webresults:
       response = webresults.read().decode('utf-8')
    
    if response == '':
        print('No UniProt mapping results found...\nNo dataset created.')
        return pd.DataFrame(), pd.DataFrame()
    else:
        df_uniprot = pd.read_csv(StringIO(response), sep='\t', dtype=str)
        entrez_list = df_uniprot.columns.tolist()[-1]
        df_uniprot.rename(columns={entrez_list: 'EntrezGeneID', 'Entry': 'ProteinID'}, inplace=True)
        # Remove unreviewed entries
        df_uniprot = df_uniprot[df_uniprot['Status'] == 'reviewed']
        df_uniprot.reset_index(inplace=True, drop=True)
        df_uniprot = df_uniprot.drop(columns=['Status'])
        
        # Map IDs to BioGRID dataset, remove unmapped genes, and rename columns
        mapped = df_biogrid.copy()
        refdict = pd.Series(df_uniprot['ProteinID'].values, index=df_uniprot['EntrezGeneID']).to_dict()
        mapped['Entrez Gene Interactor A'] = mapped['Entrez Gene Interactor A'].map(refdict)
        mapped['Entrez Gene Interactor B'] = mapped['Entrez Gene Interactor B'].map(refdict)
        mapped.dropna(subset=['Entrez Gene Interactor A', 'Entrez Gene Interactor B'], inplace=True)
        mapped.rename(columns={'Entrez Gene Interactor A':'Protein A', 'Entrez Gene Interactor B':'Protein B'}, inplace=True)
        mapped.reset_index(inplace=True, drop=True)
        
        # Repeat for protein sequences mapping for .fasta
        proteins = pd.Series(mapped['Protein A'].append(mapped['Protein B']).unique(), name='ProteinID')
        sequences = pd.Series(mapped['Protein A'].append(mapped['Protein B']).unique(), name='Sequence')
        refdictseq = pd.Series(df_uniprot['Sequence'].values, index=df_uniprot['ProteinID']).to_dict()
        fasta = pd.DataFrame(proteins).join(sequences)
        fasta['Sequence'] = fasta['Sequence'].map(refdictseq)
        fasta.dropna(inplace=True)
        fasta.drop_duplicates(inplace=True)
        fasta = fasta.reset_index(drop=True)
        
        # Drop all mapped interactions containing proteins with no sequence
        proteins_with_sequence = fasta['ProteinID']
        mapped = mapped[mapped['Protein A'].isin(proteins_with_sequence)]
        mapped = mapped[mapped['Protein B'].isin(proteins_with_sequence)]
        mapped.dropna(inplace=True)
        mapped.drop_duplicates(inplace=True)
        mapped = mapped.reset_index(drop=True)
        
        # Write out .tsv interactions and .fasta sequences, note: overwrites files of same name
        if not mapped.empty:
            if not os.path.isfile(path + organismRelease + '_ID_'  + organisms_name + '_positive_interactions.tsv'):
                print('Creating positive interactions for organismID:', organisms_name)
                mapped.to_csv(path + organismRelease + '_ID_'  + organisms_name + '_positive_interactions.tsv', columns=['Protein A', 'Protein B'], sep='\t', header=False, index=False)
            else:
                print(path + organismRelease + '_ID_'  + organisms_name + '_positive_interactions.tsv', 'already exists.')
        else:
            print('No positive interactions created for organismID:', organisms_name, 'in', organismRelease, '...check geneID-proteinID mappings...')
        
        if not fasta.empty:
            if not os.path.isfile(path + organismRelease + '_ID_'  + organisms_name + '_positive_sequences.fasta'):
                print('Creating positive sequences for organismID:', organisms_name)
                fasta_copy = fasta.copy()
                fasta_copy['ProteinID'] = ('>' + fasta_copy['ProteinID'])
                fasta_copy.to_csv(path + organismRelease + '_ID_'  + organisms_name + '_positive_sequences.fasta', columns=['ProteinID', 'Sequence'], sep='\n', header=False, index=False)
            else:
                print(path + organismRelease + '_ID_'  + organisms_name + '_positive_sequences.fasta', 'already exists.')
        else:
            print('No positive sequences created for organismID:', organisms_name, 'in', organismRelease, '...check proteinID-sequence mappings...')
        return mapped, fasta
    
def verify_sequences(mapped, fasta, proteome, organismIDs):
    filepath = PROTEOMEPATH + organismIDs + '_non-reference-proteome_appended.fasta'
    # Verify all mapped proteins have sequences in the proteome
    if not mapped.empty and not fasta.empty:
        proteins = pd.Series(mapped['Protein A'].append(mapped['Protein B']).unique())
        sequenced = pd.Series(proteome['Entry'].unique())
        found = proteins.isin(sequenced)
        # Confirm all proteins are found in proteome, give warning if not
        if len(found[found == True].index) != len(proteins):
            print('Warning!\n\tProteins exist in positive dataset that do not exist in proteome!')
            print('\tThere are', len(proteins), 'proteins,', len(found[found == True].index), 'are found in proteome file...')
            indices = found[found == False].index
            no_proteome = proteins.iloc[indices]
            not_found = fasta.iloc[no_proteome.index]
            not_found = not_found.rename(columns={'ProteinID': 'Entry'})
            appended_proteome = proteome.append(not_found, ignore_index=True, sort=False)
            appended_proteome['Entry'] = ('>' + appended_proteome['Entry'])
            appended_proteome.to_csv(filepath, columns=['Entry', 'Sequence'], sep='\n', header=False, index=False)
            print(len(found[found == False].index), 'interacting proteins not found in proteome have been added to separate file...')

if __name__ == "__main__":
    
    # Get BioGRID files from supplied directory path
    try:
        path_to_biogrid_files = sys.argv[1]
        files = os.listdir(path=path_to_biogrid_files)
        print('Using files:\n',files)
    except Exception as e:
        print(e, "\nPlease provide path to BioGRID .txt files, for example: 'python get_biogrid_interactions.py myFiles/'")
        exit()
        
    # Create directories to store data
    if not os.path.exists(INTRAPATH):
        os.mkdir(INTRAPATH)
    if not os.path.exists(INTERPATH):
        os.mkdir(INTERPATH)
    if not os.path.exists(PROTEOMEPATH):
        os.mkdir(PROTEOMEPATH)
    
    # Collect and filter BioGRID interactions
    bgframes = get_biogrid_data(path_to_biogrid_files, files, filters=True)
    
    # Iterate over each BIOGRID-ORGANISM release file
    for organismRelease, df_biogrid in bgframes.items():
        print('\n=== Working on', organismRelease, '===')
        
        # Separate intra/inter-species interactions, return lists of dataframes
        intra_interactions, inter_interactions = separate_species_interactions(organismRelease, df_biogrid)
        
        # Map intra-species interactions and write to files
        if len(intra_interactions) > 0:
            for df in intra_interactions:
                mapped, fasta = map_bgup(organismRelease, df)
                if not mapped.empty and not fasta.empty:
                    organismID = df[ORGANISM_ID_A].append(df[ORGANISM_ID_B]).unique().astype(str)
                    proteome = get_organism_proteome(organismID[0], reviewed='yes')
                    verify_sequences(mapped, fasta, proteome, organismID[0])
        else:
            print('No intraspecies interactions')
        # Map inter-species interactions and write to files
        if len(inter_interactions) > 0:
            for df in inter_interactions:
                mapped, fasta = map_bgup(organismRelease, df)
                if not mapped.empty and not fasta.empty:
                    organismID = df[ORGANISM_ID_A].append(df[ORGANISM_ID_B]).unique().astype(str)
                    proteome_a = get_organism_proteome(organismID[0], reviewed='yes')
                    proteome_b = get_organism_proteome(organismID[1], reviewed='yes')
                    full_sequences = proteome_a.append(proteome_b).drop_duplicates()
                    full_sequences.reset_index(drop=True, inplace=True)
                    verify_sequences(mapped, fasta, full_sequences, organismID[0]+'-'+organismID[1])
        else:
            print('No interspecies interactions')