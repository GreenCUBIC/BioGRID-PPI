#!/bin/bash
proteins=$(ls -A1 dataTrain/)
for p in $proteins
do
	psiblast -db swissprot -evalue 0.001 -query Proteins/${p} -out_ascii_pssm dataTrain/${p} -out dataTrain/${p}_output_file -num_iterations 3
	cat dataTrain/${p} | sed '1,3d' | tac | sed '1,6d' | tac | cut -d' ' -f8- | column -t | tr ' ' '\t' | tr -s '\t' > dataTrain/${p}.mtx
	rm dataTrain/${p}_output_file
	rm dataTrain/${p}
done
cd dataTrain/
find . -depth -name "*.txt" -exec sh -c 'f="{}"; mv -- "$f" "${f%.txt}"' \;
