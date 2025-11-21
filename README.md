Paper: "TrambaHLApan: A Transformer and Mamba-based Neoantigen Prediction Method Considering both Antigen Presentation and Immunogenicity" (DOI: 10.1007/s12539-025-00777-5)

Published in Interdiscip Sci Comput Life Sci: https://link.springer.com/article/10.1007/s12539-025-00777-5


# TrambaHLApan
TrambaHLApan, a novel neoantigen prediction framework that integrates Transformer and Mamba architectures to concurrently predict antigen presentation likelihood (TrambaHLApan-EL) and immunogenic potential (TrambaHLApan-IM). Experimental results on independent datasets demonstrate that TrambaHLApan outperforms state-of-the-art methods, establishing it as a reliable tool for advancing personalized cancer immunotherapies.

This repository contains the data and python script in support of the manuscript: TrambaHLApan: a transformer and mamba-based neoantigen prediction method considering both antigen presentation and immunogenicity.
_ _ _ _

## 1.  Requirement

#### 1) System requirements
This tool is supported for Linux. The tool has been tested on the following system:

+ Ubuntu 18.04

#### 2) Hardware requirements
We ran the demo using the following specs:

+ NVIDIA RTX 4060Ti
+ CUDA 11.8

#### 3) Mamba
In order to run optimized [Mamba](https://github.com/state-spaces/mamba) implementations, you need to install mamba-ssm and causal-conv1d:

```console
pip install causal-conv1d>=1.3.0
pip install mamba-ssm
```

#### 4) Dependencies

```console
python==3.8.19
pytorch == 2.0.0
torchvision==0.15.0
numpy==1.23.5
pandas==2.0.3
scikit-learn==1.3.0
tqdm==4.65.0
matplotlib=3.7.2
weblogo==3.7.12
```

## 2.  How to train and use TrambaHLApan

#### 1) Download all data from the Mendeley Data repository (https://data.mendeley.com/datasets/kctz3mrwgz/2) and place the data in the `./data/datasets/` folder.

#### 2) Training the TrambaHLApan-EL model

```console
python  main_train_Model_EL.py   --fold 0   --index 0
python  main_train_Model_EL.py   --fold 1   --index 0
python  main_train_Model_EL.py   --fold 2   --index 0
python  main_train_Model_EL.py   --fold 3   --index 0
python  main_train_Model_EL.py   --fold 4   --index 0
```

After the training, the trained TrambaHLApan-EL models will be saved in the `./output/models/` folder.

#### 3) Training the TrambaHLApan-IM model

```console
python  main_train_Model_IM.py   --fold 0   --index 0
python  main_train_Model_IM.py   --fold 1   --index 0
python  main_train_Model_IM.py   --fold 2   --index 0
python  main_train_Model_IM.py   --fold 3   --index 0
python  main_train_Model_IM.py   --fold 4   --index 0
```

After the training, the trained TrambaHLApan-IM models will be saved in the `./output/models/` folder.

#### 4) Make predictions for the test datasets

```console
python TrambaHLApan_predict.py --input ./data/datasets/DataS3.csv --output ./output/results/DataS3_by_TrambaHLApan.csv
python TrambaHLApan_predict.py --input ./data/datasets/DataS4.csv --output ./output/results/DataS4_by_TrambaHLApan.csv
python TrambaHLApan_predict.py --input ./data/datasets/DataS5.csv --output ./output/results/DataS5_by_TrambaHLApan.csv
python TrambaHLApan_predict.py --input ./data/datasets/DataS6.csv --output ./output/results/DataS6_by_TrambaHLApan.csv
```
+ --input: the path of the test datasets.
+ --output: the path where you want to save the prediction results.
+ You can refer to `demo.csv` to prepare your input information. the first column is the sequence of the peptide, and the second column is the name of the HLA.
+ The file `./data/datasets/pseudoseqs.csv` records the pseudo sequences of all HLA molecules used in this study. For other HLA molecules, their pseudo sequences can be found in the file `./data/NetMHCpan4.1/MHC_pseudo.dat`.
 
After the running, the prediction results will be saved in the `./output/results/` folder.

#### 5) Obtain the immune epitope motif for the specified HLA：
  
```console
python TrambaHLApan_motif.py --HLA HLA-A*02:01 --HLAseq YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY
python TrambaHLApan_motif.py --HLA HLA-A*11:01 --HLAseq YYAMYQENVAQTDVDTLYIIYRDYTWAAQAYRWY
python TrambaHLApan_motif.py --HLA HLA-B*07:02 --HLAseq YYSEYRNIYAQTDESNLYLSYDYYTWAERAYEWY
python TrambaHLApan_motif.py --HLA HLA-B*15:02 --HLAseq YYAMYRNISTNTYESNLYIRYDSYTWAELAYLWY
python TrambaHLApan_motif.py --HLA HLA-B*27:05 --HLAseq YHTEYREICAKTDEDTLYLNYHDYTWAVLAYEWY
python TrambaHLApan_motif.py --HLA HLA-B*40:02 --HLAseq YHTKYREISTNTYESNLYLSYNYYTWAVLAYEWY
```

+ --HLA: the name of the HLA molecule
+ --HLAseq: the pseudo-sequence of the HLA molecule

## 3. Model availability
Trained TrambaHLApan models on the comprehensive training datasets are available on the `./output/models/` folder. And you can use it on your own dataset.

## 4. Citation
Please cite our paper if you use the code.

## 5. Contact
If you have any questions, please contact us via email: 
+ [Xiumin Shi](sxm@bit.edu.cn)



