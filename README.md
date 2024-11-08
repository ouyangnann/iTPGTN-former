# Traffic Prediction

This repository contains the code for the paper **"Graph Transformer-based Dynamic Edge Interaction Encoding for Traffic Prediction"**.

## Table of Contents
- [Introduction](#introduction)
- [Environment](#environment)
- [Installation](#installation)
- [Usage](#usage)
  - [Dataset Preparation](#dataset-preparation)
  - [Training and Testing](#training-and-testing)

## Introduction
We propose a dynamic edge interaction encoding method for spatio-temporal features based on inverse Transformer (iTransformer) and Graph Transformer, named iTPGTN-former.

## Environment
The code is developed and tested on **Ubuntu 20.04**. The required libraries and their versions are as follows:

- **Python Libraries**
  - `torch==2.1.0+cu121`
  - `torchaudio==2.1.0+cu121`
  - `torchvision==0.16.0+cu121`
  - `tensorboard==2.14.0`
  - `dgl==2.4.0+cu118`
  - `iTransformer`
  - `matplotlib`
  - `tables`

## Installation
To set up the environment, you can use the following steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ouyangnann/iTPGTN-former.git
   cd iTPGTN-former
   ```
   
2. **Install required packages**:

   Create a conda environment with python 3.8:
   ```bash
   conda create --name itpgtn python=3.8
   ```
   
   Install all required libraries using `pip`:
   ```bash
   conda activate itpgtn
   pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
   pip install tensorboard
   pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu118/repo.html
   pip install iTransformer
   pip install matplotlib
   pip install tables
   ```
  
## Usage
 ## Dataset Preparation:
  Download the required datasets. See data/readme.md.

  Prepare datasets using the following scripts:
  
  MATE-LA and PEMS-04:
  ```bash
  python scripts/generate_training_data_metr_in_ou.py --seq_len 12 --horizon 12
  python scripts/gen_adj_mx.py 
  ```
  
  PEMS03, PEMS04, PEMS07, and PEMS08:
  
  ```bash
  python scripts/generate_training_data_pems0408_in_ou.py --seq_len 12 --horizon 12
  python scripts/gen_adj_mx0408.py 
  ```
  The seq_len and horizon are optional.

 ## Training and Testing:
  Train the model: Run the run.sh script to start training:
  ```bash
  sh run.sh
  ```
  
  Test the model: Use the test.sh script to evaluate the trained model:
  
  ```bash
  sh test.sh
  ```
