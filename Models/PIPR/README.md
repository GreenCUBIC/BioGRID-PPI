PIPR
Muhao Chen, Chelsea J -T Ju, Guangyu Zhou, Xuelu Chen, Tianran Zhang, Kai-Wei Chang, Carlo Zaniolo, Wei Wang, Multifaceted protein–protein interaction prediction based on Siamese residual RCNN, Bioinformatics, Volume 35, Issue 14, July 2019, Pages i305–i314, https://doi.org/10.1093/bioinformatics/btz328

Usage:  

e.g.  
CUDA_VISIBLE_DEVICES=0 python pipr_rcnn.py all_sequences.fasta dataTrain.tsv dataTest.tsv  

Sequences file must contain all protein IDs and sequences from train and test data in .tsv format, eg:  
PROTEINA  SEQUENCEASLSFPVTSSMVSSTSSYSSFLFLLVVNHLFSGRLRCGSPEFIIRSFTITLGPLNHNISPFVFFH  
PROTEINB	TQMTSEQUENCEBPAPKISYKKGAASNRTKFVRSLVREIAGLSPYERRLIDLIRNSGEKRARKVAKKRLGSFTRAKAKVERH  
PROTEINC	DLATKINEKPSEQUENCECTVVNDYEAARAIPNQQVLSKLERAAPK  

Train data must contain protein IDs and labels (1=interacts, 0=does not interact) and be in .tsv with a header, eg:  
v1	v2	label  
PROTEINA	PROTEINA	1  
PROTEINA	PROTEINB	1  
PROTEINB	PROTEINC	1  
PROTEINX	PROTEINA	0  
PROTEINY	PROTEINB	0  
PROTEINA	PROTEINZ	0  

Test data must be in the same format as train data.
<i>Note: if train data and test data are the same, a 5-fold cross-validation will be performed on the provided data.</i>  


