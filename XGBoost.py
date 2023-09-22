import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from preprocessXGBoost import generateSequence
from sklearn.metrics import roc_auc_score

def predictXGBoost(model_path, test_path, test_filename, result_path, variant):

    generateSequence(test_path, test_filename)
    test_seq_path = test_path + "data_poiFeat.csv"
    result_model = model_path + "model.txt"
    result_prediction = result_path + "prediction.csv"

    test = pd.read_csv(test_path + test_filename)

    # Check if "significance" column exists in the dataframe
    has_significance_col = "significance" in test.columns

    if has_significance_col:
        y_test = test[["significance"]].values
    else:
        y_test = None

    X_test2 = pd.read_csv(test_seq_path)

    if variant == "seq_anno":
        X_test = test[['OGEE_prop_Essential', 'deltagb', 'deltagh', "H3k27ac_CPM_1Kb_new", "ATAC_CPM_1Kb_new", "H3K4me3_CPM_1Kb_new"]]
        X_test3 = pd.concat([X_test, X_test2], axis=1)
    else:
        X_test3 = X_test2

    print("There are ", X_test3.shape[0], "gRNAs in test data.")

    dtest = xgb.DMatrix(X_test3, label=y_test)
    best_model = xgb.Booster()
    best_model.load_model(result_model)
    
    test_est = best_model.predict(dtest)

    if has_significance_col:
        print('\n> AUC score is', roc_auc_score(y_test, test_est))
        PD = pd.DataFrame(np.column_stack((test['protospacer'], y_test, test_est)), columns=['grna', 'true', 'predict'])
    else:
        PD = pd.DataFrame(np.column_stack((test['protospacer'], test_est)), columns=['grna', 'predict'])

    PD.to_csv(result_prediction, index=False)
    print('\n> Prediction data saved in result path! \n')



def trainXGBoost(model_path, train_path, train_filename, variant, params):

    # Assuming you have defined the generateSequence function somewhere
    generateSequence(train_path, train_filename)

    train_seq_path = train_path + "data_poiFeat.csv"
    result_model = model_path + "model.txt"

    train = pd.read_csv(train_path+train_filename)
    y = train[["significance"]].values
    X2 = pd.read_csv(train_seq_path)

    if variant == "seq_anno":
        X = train[['OGEE_prop_Essential', 'deltagb','deltagh', "H3k27ac_CPM_1Kb_new", "ATAC_CPM_1Kb_new","H3K4me3_CPM_1Kb_new"]]
        X_train = pd.concat([X, X2], axis=1)
    else:
        X_train = X2

    print("There are ", X_train.shape[0], "gRNAs and ", X_train.shape[1], "features to train.")

    # Splitting your data into a training set and a validation set
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y, test_size=0.2, random_state=42)

    # Converting datasets into DMatrix format for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y,enable_categorical=True)
    dtrain_split = xgb.DMatrix(X_train_split, label=y_train_split, enable_categorical=True)
    dval = xgb.DMatrix(X_val_split, label=y_val_split, enable_categorical=True)

    num_boost_round = 999
    evals_result = {}  # This will store the evaluation results for each round
    print('\n> Splitting train data into a training set and a validation set to select the best number of rounds \n')
    model = xgb.train(
        params,
        dtrain_split,
        num_boost_round= num_boost_round,  # a large number, early stopping will determine the actual best number
        evals=[(dtrain_split, "Train"), (dval, "Validation")],
        early_stopping_rounds=10,
        evals_result=evals_result,
        verbose_eval=True
    )
    # Best number of rounds based on validation set performance
    best_rounds = model.best_iteration

    print(f"Best number of rounds based on validation set: {best_rounds}")
    print('\n> Train a final model on the entire dataset using the best number of rounds \n')
    # Now, if you want, you can train a final model on the entire dataset using the best number of rounds
    final_model = xgb.train(params, dtrain, num_boost_round=best_rounds, verbose_eval=True)

    final_model.save_model(result_model)
    print('\n> Model Saved! \n')