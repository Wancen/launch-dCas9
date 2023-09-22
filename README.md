# launch-dCas9
launch-dCas9 provides machine LeArning based UNified CompreHensive framework to predict gRNA impact from multiple perspectives, including cell fitness, wild-type abundance (gauging power potential), and gene expression in single cells.

launch-dCas9 provides functions including derive testing data predictions according to pre-trained model or derive trained model given training datasets.

# Input files

* Sequencing model - performs well on cell fitness task for gRNAs in enhancer regions and wild type counts task

* Integrating Sequencing and annotation model - performs well on cell fitness task for gRNAs in enhancer regions and wild type counts task
We accept csv with fixed column names ["protospacer",'OGEE_prop_Essential', 'deltagb','deltagh', "H3k27ac", "ATAC", "H3K4me3"] as either training data or testing data.
The annotation columns have to be continuous value. launch-dCas9 use zero-imputation internally.

# Evaluation
If you want to access the prediction performance in testing data, the outcome column have to be named "significance" in the testing data.

# Installation
Start by grabbing this source codes:
```bash
git clone https://github.com/Wancen/launch-dCas9.git
cd launch-dCas9
```
Use python virutal environment with conda
```bash
conda env create -f environment.yml
```
or 
```bash
conda create -n launch-dCas9 --file requirements.txt
conda activate launch-dCas9
```

# Usage
## Prediction 
Have to specify *model_path*, *test_path*, *test_filename*, *result_path*, *outcome*
* CNN

```bash
python -W ignore launch-dCas9.py \
    --model CNN \
    --model_path ./exampleData/ \
    --test_path ./exampleData/ \
    --test_filename test.csv\
    --result_path ./exampleData/ \
    --variant seq_anno \
    --outcome promoterFitness

```

* XGBoost

```bash
python -W ignore launch-dCas9.py \
    --model XGBoost \
    --model_path ./exampleData/ \
    --test_path ./exampleData/ \
    --test_filename test.csv\
    --result_path ./exampleData/ \
    --variant seq_anno \
    --outcome promoterFitness
```

## Get trained model
Have to specify *model_path*, *train_path*, *train_filename*, *outcome*

* CNN

```bash
python -W ignore launch-dCas9.py \
    --model CNN \
    --model_path ./exampleData/ \
    --train_path /proj/milovelab/mu/dukeproj/data/dat_discovery/promoter/ \
    --train_filename wgCERES-gRNAs-k562-discovery-screen-pro_baseMean125-binary-1-train.csv\
    --variant seq_anno\
    --outcome promoterFitness
```

* XGBoost 

```bash
python -W ignore launch-dCas9.py \
    --model XGBoost \
    --model_path ./exampleData/ \
    --train_path /proj/milovelab/mu/dukeproj/data/dat_discovery/promoter/ \
    --train_filename wgCERES-gRNAs-k562-discovery-screen-pro_baseMean125-binary-1-train.csv\
    --variant seq_anno\
    --outcome promoterFitness
```