#!/bin/bash

python -W ignore launch-dCas9.py \
    --model XGBoost \
    --model_path ./exampleData/ \
    --test_path ./exampleData/ \
    --test_filename test.csv\
    --result_path ./exampleData/ \
    --variant seq_anno
# train the training dataset to get model.txt for XGBoost model
# python -W ignore launch-dCas9.py \
#     --model XGBoost \
#     --model_path ./exampleData/ \
#     --train_path /proj/milovelab/mu/dukeproj/data/dat_discovery/promoter/ \
#     --train_filename wgCERES-gRNAs-k562-discovery-screen-pro_baseMean125-binary-1-train.csv\
#     --variant seq_anno\
#     --outcome promoterFitness

# predict testing data result based on model.txt

    