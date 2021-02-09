**[DPPI](https://github.com/hashemifar/DPPI)**  
Hashemifar S, Neyshabur B, Khan AA, Xu J. Predicting protein-protein interactions through sequence-based deep learning. Bioinformatics. 2018 Sep 1;34(17):i802-i810. doi: 10.1093/bioinformatics/bty573. PMID: 30423091; PMCID: PMC6129267. 
 
___
## Usage:  

e.g.  
> **th dppi.lua -train dataTrain -test dataTest -device 1**  

- First arg supplies name for training data, second arg supplies name for test data  
- Each dataset should contain the following using the same name  
e.g.  
1. A **dataTrain.node** file containing a list of all protein IDs  
> PROTEIN1  
> PROTEIN2  
> PROTEIN3  
> ...  
2. A **dataTrain.csv** file containing the interactions with labels  
> PROTEIN1,PROTEIN2,1  
> PROTEIN1,PROTEIN1,1  
> PROTEIN3,PROTEIN2,1  
> PROTEIN1,PROTEIN3,0  
> PROTEIN2,PROTEIN2,0  
> ...  
3. A  **dataTrain/** directory containing files named as the protein IDs found in the .node file without any extension.
Each file's contents contain the PSSM results table of a BLAST search for that protein against a database


<i>Note 1: if train data and test data args are the same, a 5-fold cross-validation will **NOT** be performed on the provided data.</i> 

### Requirements:  
lua
cuda
torch
