# BioGRID Dataset Processing

After downloading and unzipping a [BioGRID release](https://downloads.thebiogrid.org/BioGRID/Release-Archive/) file, run the following:

1. get_biogrid_interactions.py <path_to_.tab2_or_.tab3_files> -f -m  
    
   This generates three directories with files with filtered positive protein interactions related to organismIDs and files containing their sequences:  
   * Intraspecies_Interactions/  
   * Interspecies_Interactions/  
   * Proteomes/  
   
   Option -f will apply filters to interactions  
   Option -m will check for multiple different publication sources of interactions  
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
   
2. generate_negative_interactions.py -pseq <file_positive_sequences.fasta> -p <file_positive_interactions.tsv> -d  
   
   This will generate negative interaction data based on random sampling of proteins in the positive interactions, not found in positive pairs.  
   Option -d will generate the random pairs having no similar subcellular locations as listed by UniProt  
   *Currently only supports negatives for Intraspecies_Interactions/.*  
    
3. verify_balance_interactions.py -pseq <file_positive_sequences.fasta> -p <file_positive_interactions.tsv> -nseq <file_negative_sequences.fasta> -n <file_negative_interactions.tsv> -output <file_suffix_string>  

   This may be run on any dataset, after homology reductions, or not at all. It ensures that all proteins found in interactions have corresponding sequences and will remove interactions if there is no sequence for either or both proteins. It also makes sure the positive and negative interaction data is balanced.  
    
4. format_interactions.py -pseq <file_positive_sequences.fasta> -p <file_positive_interactions.tsv> -nseq <file_negative_sequences.fasta> -n <file_negative_interactions.tsv> -m <model_name_from_options>  

   This will generate new directories for the chosen models/formats and format the provided data to the corresponding directory.  
   *Currently only supports formatting for "PIPR", "DeepFE-PPI", "DPPI", and "SPRINT" protein prediction models.*  
   *Note: DPPI will still require a PSI-BLAST search for PSSMs using the protein files created.*  
   [PIPR](https://github.com/muhaochen/seq_ppi)  
   [DeepFE-PPI](https://github.com/xal2019/DeepFE-PPI)  
   [DPPI](https://github.com/hashemifar/DPPI)  
   [SPRINT](https://github.com/lucian-ilie/SPRINT)  


#### Positive interactions are filtered based on conservative approach prescribed by [POSITOME](http://bioinf.sce.carleton.ca/POSITOME/)  
K. Dick, F. Dehne, A. Golshani and J. R. Green, "Positome: A method for improving protein-protein interaction quality and prediction accuracy," 2017 IEEE Conference on Computational Intelligence in Bioinformatics and Computational Biology (CIBCB), Manchester, 2017, pp. 1-8, doi: 10.1109/CIBCB.2017.8058545.  

