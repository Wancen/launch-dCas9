# launch-dCas9
launch-dCas9 provides machine LeArning based UNified CompreHensive framework to predict gRNA impact from multiple perspectives, including cell fitness, wild-type abundance (gauging power potential), and gene expression in single cells.

# Preprocess input files
We accept csv with fixed column names ["protospacer",'OGEE_prop_Essential', 'deltagb','deltagh', "H3k27ac", "ATAC", "H3K4me3"] as either training data or testing data.
The annotation columns have to be continuous value.

If you want to access the prediction performance in testing data, the outcome column have to be named "significance" in the testing data.

# Installation

# Usage
## Prediction 
* XGBoost

```bash
python -W ignore launch-dCas9.py \
    --model XGBoost \
    --model_path ./exampleData/ \
    --test_path ./exampleData/ \
    --test_filename test.csv\
    --result_path ./exampleData/ \
    --variant seq_anno
```

## Get trained model
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