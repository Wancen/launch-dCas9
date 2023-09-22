import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from util import preprocess_seq
import model
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch
import torchvision
import torch.nn as nn


def trainCNN(model_path, train_path, train_filename, variant, params, outcome):
    train = pd.read_csv(train_path+train_filename, index_col = False)
    sequence = train['protospacer']
    # generate onehot encoding of protospacer and position feature dataframe
    sequence_onehot, position_feat = preprocess_seq(sequence)
    label = train['significance'].to_numpy(dtype = np.float32)
    class_count = train['significance'].value_counts()
    w = class_count[0] / class_count[1]

    X1 = torch.tensor(sequence_onehot, dtype=torch.float32)
    Y = torch.tensor(label, dtype=torch.float32)
    Y = Y.view(-1, 1)

    if variant == "seq_anno":
        '''Top features that we keep: deltagb, deltagh, H3K27ac, ATAC, H3K4me3, OGEE_prop_Essential'''
        anno_sel = ["deltagb", "deltagh", "OGEE_prop_Essential", "H3k27ac_CPM_1Kb_new", "ATAC_CPM_1Kb_new", "H3K4me3_CPM_1Kb_new"]
        annotation_df = train[anno_sel]

        # Concatenate position_feat and annotation_df along the columns axis
        combined_df = pd.concat([position_feat, annotation_df], axis=1)
        combined_df = combined_df.fillna(0)
        # If you need the result as a NumPy array
        annotation = combined_df.to_numpy(dtype=np.float32)
        X2 = torch.tensor(annotation, dtype=torch.float32)
        input_dat = TensorDataset(X1,X2,Y)
    else:
        input_dat = TensorDataset(X1,Y)

    batch_size = params['batch_size']

    device = params['device']
    datloader = DataLoader(input_dat, batch_size=batch_size, shuffle=True)

    CNN = model.DeepSeqCNN(dim_y=combined_df.shape[1] if variant == "seq_anno" else 0).to(device)

    # Calling the train_model function with the appropriate parameters
    CNN = model.train_model(CNN, datloader, variant=variant, params = params, weight = w)
    ckptPATH = model_path + 'ckpt-'+ variant + '-'+ outcome +'.pth'
    torch.save(CNN.state_dict(), ckptPATH)


def predictCNN(model_path, test_path, test_filename, result_path, variant, outcome):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_prediction = result_path + "prediction-"+ variant +'-'+ outcome + ".csv"
    test = pd.read_csv(test_path + test_filename)
    test_sequence = test['protospacer']
    test_sequence_onehot, test_position_feat = preprocess_seq(test_sequence)
    test_X1 = torch.tensor(test_sequence_onehot, dtype=torch.float32).to(device)
    
    # Check if "significance" column exists in the dataframe
    has_significance_col = "significance" in test.columns
    if has_significance_col:
        test_label = test['significance'].to_numpy(dtype = np.float32)
    else:
        y_test = None

    if variant == "seq_anno":
        '''Top features that we keep: deltagb, deltagh, H3K27ac, ATAC, H3K4me3, OGEE_prop_Essential'''
        anno_sel = ["deltagb", "deltagh", "OGEE_prop_Essential", "H3k27ac_CPM_1Kb_new", "ATAC_CPM_1Kb_new", "H3K4me3_CPM_1Kb_new"]
        annotation_df = test[anno_sel]

        # Concatenate position_feat and annotation_df along the columns axis
        combined_df = pd.concat([test_position_feat, annotation_df], axis=1)
        combined_df = combined_df.fillna(0)
        # If you need the result as a NumPy array
        test_annotation = combined_df.to_numpy(dtype=np.float32)
        test_X2 = torch.tensor(test_annotation, dtype=torch.float32).to(device)
    
    # Define the path to the checkpoint file
    ckpt_path = model_path + 'ckpt-' + variant + '-' + outcome + '.pth'

    # Create an instance of your model (assuming your model is an instance of DeepSeqCNN)
    CNN = model.DeepSeqCNN(dim_y=combined_df.shape[1] if variant == "seq_anno" else 0)
    CNN = CNN.to(device)
    # Load the checkpoint
    checkpoint = torch.load(ckpt_path)

    # Load the model state_dict from the checkpoint
    CNN.load_state_dict(checkpoint)

    # Ensure the model is in evaluation mode (if needed)
    CNN.eval()

    sigmoid = nn.Sigmoid()
    if variant == "seq_anno":
        test_predict = sigmoid(CNN(test_X1, test_X2, variant = variant))
    else:
        test_predict = sigmoid(CNN(test_X1, variant = variant))

    test_predict_np = test_predict.detach().to('cpu').numpy()
    if has_significance_col:
        print('\n> AUC score is', roc_auc_score(y_test, test_est))
        PD = pd.DataFrame(np.column_stack((test['protospacer'], test_label, test_predict_np[:,0])), columns=['grna', 'true', 'predict'])
    else:
        PD = pd.DataFrame(np.column_stack((test['protospacer'], test_predict_np[:,0])), columns=['grna', 'predict'])

    PD.to_csv(result_prediction, index=False)
    print('\n> Prediction data saved in result path! \n')








