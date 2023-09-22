import argparse
import pandas as pd
import numpy as np
from CNN import trainCNN
from XGBoost import trainXGBoost, predictXGBoost
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and predict with launch-dCas9.')
    
    parser.add_argument('--model', type=str, help='Model to train', default="CNN", choices=["CNN", "XGBoost"])
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model.')
    parser.add_argument('--test_path', type=str, required=False, help='Path to the testing data.')
    parser.add_argument('--test_filename', type=str, required=False, help='Filename of the testing data.')
    parser.add_argument('--train_path', type=str, required=False, help='Path to the training data.')
    parser.add_argument('--train_filename', type=str, required=False, help='Filename of the training data.')
    parser.add_argument('--result_path', type=str, required=False, help='Path to save the results.')
    parser.add_argument('--variant', type=str, help='Model variants to train', default="seq_anno", choices=["seq_anno", "seq"])
    parser.add_argument('--outcome', type=str, help='Pertubation outcomes to train', default="single-cell", choices=["promoterFitness", "enhancerFitness","single-cell","WTcounts"])

    # CNN parameters
    parser.add_argument('--batch_size', type=int, default=256, help='(int, default 256) Batch size')
    parser.add_argument('--epochs', type=int, help='(int, default 60 for promoterFitness, 15 for enhancerFitness) The epoch for CNN')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

    # XGBoost parameters
    parser.add_argument('--max_depth', type=int, help='Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.')
    parser.add_argument('--min_child_weight', type=float, help='Minimum sum of instance weight (hessian) needed in a child. Used to control overfitting. Higher values prevent more partitioning, resulting in more conservative models.')
    parser.add_argument('--eta', type=float, help='Step size shrinkage used in update to prevent overfitting. After each boosting step, we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative.')
    parser.add_argument('--subsample', type=float, help='Proportion of training data to randomly sample in each round to prevent overfitting. Setting it to 0.5 means that XGBoost randomly samples half of the training data prior to growing trees, which prevents overfitting.')
    parser.add_argument('--colsample_bytree', type=float, help='Proportion of columns (features) to be randomly sampled for building each tree.')
    parser.add_argument('--reg_alpha', type=float, help='L1 regularization term on weights. Increasing this value makes models more conservative.')
    parser.add_argument('--scale_pos_weight', type=float, help='Controls the balance of positive and negative weights, useful for highly imbalanced classes. A typical value to consider: sum(negative instances) / sum(positive instances).')
    parser.add_argument('--objective', type=str, help='Specifies the learning task and the corresponding objective function (e.g., "reg:squarederror" for regression tasks, "binary:logistic" for binary classification).')
    parser.add_argument('--eval_metric', type=str, help='Evaluation metric to be used for validation data (e.g., "rmse" for regression tasks, "auc" for classification error rate).')

    args = parser.parse_args()
    print('\n> Loading Packages')
    if args.train_path is not None:
        print('\n> Training data ...')
        if args.model == "CNN":
            if args.outcome == "enhancerFitness":
                epochs = 15
            elif args.outcome == "promoterFitness":
                epochs = 60
            else:
                print("Invalid group: " + grp)
            params = {
                    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    'epochs': epochs,
                    'lr': args.lr,
                    'batch_size': args.batch_size
                }
            print( f"Using device: {params['device']}" )
            if args.epochs:
                params['epochs'] = args.epochs

            print(args)
            trainCNN(model_path=args.model_path, 
                     train_path=args.train_path, 
                     train_filename=args.train_filename, 
                     variant=args.variant,
                     params = params)
        else:
            # Set default params based on region
            if args.outcome == "promoterFitness":
                params = {
                    'max_depth': 3,
                    'min_child_weight': 8,
                    'eta': .1,
                    'subsample': 1.0,
                    'colsample_bytree': 0.5,
                    'reg_alpha': 30,
                    'scale_pos_weight': 5,
                    "objective": "binary:logistic",
                    'eval_metric': 'auc'
                }
            elif args.outcome == "enhancerFitness":
                params = {
                    'max_depth': 8,
                    'min_child_weight': 16,
                    'eta': .1,
                    'subsample': 1.0,
                    'colsample_bytree': 0.5,
                    'reg_alpha': 1,
                    'scale_pos_weight': 5,
                    "objective": "binary:logistic",
                    'eval_metric': 'auc'
                }
            elif args.outcome == "single-cell":
                params = {
                    'max_depth': 3, 
                    'min_child_weight': 13, 
                    'eta': 0.3, 
                    'scale_pos_weight': 10, 
                    'subsample': 1.0, 
                    'colsample_bytree': 1.0, 
                    'reg_alpha': 70, 
                    'objective': 'binary:logistic', 
                    'eval_metric': 'auc'}
            elif args.outcome == "WTcounts":
                params = {
                    'max_depth': 3, 
                    'min_child_weight': 6, 
                    'eta': 0.3, 
                    'scale_pos_weight': 10, 
                    'subsample': 1, 
                    'colsample_bytree': 0.8, 
                    'reg_alpha': 30, 
                    'objective': 'reg:squarederror', 
                    'eval_metric': 'rmse'}

            # Override defaults with any specified arguments
            if args.max_depth:
                params['max_depth'] = args.max_depth
            if args.min_child_weight:
                params['min_child_weight'] = args.min_child_weight
            if args.eta:
                params['eta'] = args.eta
            if args.subsample:
                params['subsample'] = args.subsample
            if args.colsample_bytree:
                params['colsample_bytree'] = args.colsample_bytree
            if args.reg_alpha:
                params['reg_alpha'] = args.reg_alpha
            if args.scale_pos_weight:
                params['scale_pos_weight'] = args.scale_pos_weight
            if args.objective:
                params['objective'] = args.objective
            if args.eval_metric:
                params['objective'] = args.eval_metric
            print(args)
            trainXGBoost(model_path=args.model_path, 
                         train_path=args.train_path, 
                         train_filename=args.train_filename, 
                         variant=args.variant,
                         params = params)
    else:
        print('\n> Predicting data ...')
        if args.model == "CNN":
            predictCNN(model_path=args.model_path, 
                       test_path=args.test_path, 
                       test_filename=args.test_filename, 
                       result_path=args.result_path, 
                       variant=args.variant)
        else:
            print(args)
            predictXGBoost(model_path=args.model_path, 
                           test_path=args.test_path, 
                           test_filename=args.test_filename, 
                           result_path=args.result_path, 
                           variant=args.variant)

