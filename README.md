# RLEAAI
Improving Antibody-Antigen Interaction Prediction Using Protein Language Model and Sequence Order Information
## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Result](#result)
- [Contact](#contact)

## Overview
Antibody-antigen interactions (AAIs) are a pervasive phenomenon in the natural and are instrumental in the design of antibody-based drugs. Despite the emergence of various deep learning-based methods aimed at enhancing the accuracy of AAIs predictions, most of these approaches overlook the significance of sequence order information. In this study, we propose a new deep learning-based method RLEAAI, to improve the prediction performance of AAIs. In RLEAAI, a sequence order extraction strategy, called CKSAAP, is employed to generate feature representation from the feature embedding outputted by a pre-trained protein language model. In order to fully dig out the discrimination information from features, three neural network modules, i.e., convolutional neural network (CNN), bidirectional long short-term memory network (BiLSTM) and recurrent criss-cross attention mechanism (RCCA), are integrated.

## Hardware requirements

The experiments are tested and conducted on one Nvidia A100 (40GB).

## Installation

We highly recommand that you use Anaconda for Installation
```
conda create -n RLEAAI
conda activate RLEAAI
pip install -r requirements.txt
```

## Data
The HIV data and SARS-CoV-2 data is in the `data` folder.
* `data/dataset_hiv.xlsx` is the HIV data.
* `data/dataset_SARS-CoV-2.xlsx` is the SARS-CoV-2 data.

## Usage

Run an example on HIV data, please run
```
python main.py --config=configs/test_on_HIV_cls.yml
```

Run an example on SARS-CoV-2 data data, please run, please run
```
python main.py --config=configs/test_on_HIV_reg.yml
```

## Result

* The prediction result file (e.g., "3J8YK.pred") of each protein (e.g., 3J8YK) in your input fasta file (-seq_fa) could be found in the folder which you input as "-sf".
* There are four columns in each prediction result file. The 1st column is the residue index. The 2nd column is the residue type. The 3rd column is the predicted probablity of the corresponding residue belonging to the class of ATP-binding residues. The 4th column is the prediction result ('B' and 'N' mean the predicted ATP-binding and non-ATP-binding residue, respectively). For example:

~~~
Index    AA    Prob.    State
    0     A    0.001    N
    1     E    0.000    N
    2     S    0.007    N
    3     N    0.001    N
    4     I    0.000    N
    5     K    0.000    N
    6     V    0.000    N
    7     M    0.003    N
    8     C    0.000    N
    9     R    0.984    B
   10     F    0.000    N
   11     R    0.993    B
   12     P    0.990    B
   13     L    0.001    N
   14     N    0.001    N
   15     E    0.000    N
   16     S    0.005    N
   17     E    0.000    N
   18     V    0.000    N
   19     N    0.001    N
~~~


## Contact
If you have any questions, comments, or would like to report a bug, please file a Github issue.
