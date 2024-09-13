# RLEAAI
Improving Antibody-Antigen Neutralization Prediction Based on Sequence and Protein Language Model
# Dependency
-python 3.8
-pytorch 2.0.1
-biopython 1.82
-numpy 1.24.3
-scikit-learn 1.3.2
-fair-esm 2.0.0
# Usage
The script predict.py is used to predict probability of neutralization.
python predict.py --ab_fa data/ab.fasta --ag_fa data/ag.fasta
