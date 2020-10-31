# PPIP BioGRID Dataset Processing

After downloading and unzipping a BioGRID release, run the following:

1. get_biogrid_interactions.py <path_to_.tab2_or_.tab3_files>
    
   This generates three directories with files with filtered positive protein interactions related to organismIDs and files containing their sequences:  
   * Intraspecies_Interactions/  
   * Interspecies_Interactions/  
   * Proteomes/  
   
   *Note that reference proteomes will need to be downloaded separately.*  
   #### Output file format  
   
   Sequence data as .fasta files, example:  
   \>ProteinID_1  
   SEQUENCE1  
   \>ProteinID_2  
   SEQUENCE2  
   ...  
   Interaction data as .tsv files, example:  
   ProteinA  ProteinB  
   ProteinC  ProteinD  
   ...  
   
2. generate_negative_interactions.py -pseq <file_positive_sequences.fasta> -p <file_positive_interactions.tsv>  
   
   This will generate negative interaction data based on random sampling of proteins in the positive interactions, not found in positive pairs.  
   *Currently only supports negatives for Intraspecies_Interactions/.*  
    
3. verify_balance_interactions.py -pseq <file_positive_sequences.fasta> -p <file_positive_interactions.tsv> -nseq <file_negative_sequences.fasta> -n <file_negative_interactions.tsv> -output <file_suffix_string>  

   This may be run on any dataset, after homology reductions, or not at all. It ensures that all proteins found in interactions have corresponding sequences and will remove interactions if there is no sequence for either or both proteins. It also makes sure the positive and negative interaction data is balanced.  
    
4. format_interactions.py -pseq <file_positive_sequences.fasta> -p <file_positive_interactions.tsv> -nseq <file_negative_sequences.fasta> -n <file_negative_interactions.tsv> -m <model_name_from_options>  

   This will generate new directories for the chosen models/formats and format the provided data to the corresponding directory.  
   Currently only supports formatting for "PIPR", "DeepFE-PPI", "DPPI", and "SPRINT" protein prediction models.  
   Note: DPPI will still require a PSI-BLAST search for PSSMs using the protein files created.  
    
