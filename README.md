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
python predict.py --ab_fa data/example/ab.fasta --ag_fa data/example/ag.fasta --virus HIV
```

Run an example on SARS-CoV-2 data data, please run, please run
```
python predict.py --ab_fa data/example/ab.fasta --ag_fa data/example/ag.fasta --virus SARS-CoV-2
```

## Result

* The prediction result are provided in the  `results` folder.
* Each predicted results file contains two columns, with each row representing a pair of antibody and antigen samples. The first column provides the predicted probability of interaction, where a probability exceeding 0.5 indicates the presence of an interaction. The second column contains the true label, where 1.0 indicates the presence of an interaction and 0.0 indicates the absence of an interaction. For example:

~~~
Prob	    Label
0.781890	1.0
0.000014	0.0
0.492546	1.0
0.549302	0.0
0.000011	0.0
0.961460	1.0
0.364680	1.0
0.994271	1.0
0.966377	1.0
0.000141	0.0
~~~


## Contact
If you have any questions, comments, or would like to report a bug, please file a Github issue.
