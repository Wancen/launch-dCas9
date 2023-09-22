import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from util import preprocess_seq
import model
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch
import torchvision


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
    ckptPATH = model_path + 'ckpt-BCE-'+ variant + '-'+ outcome +'.pth'
    torch.save(CNN.state_dict(), ckptPATH)



def predictCNN(model_path, train_path, train_filename, variant, params):
    ## test set
    test = pd.read_csv(datadir+'/wgCERES-gRNAs-k562-discovery-screen-'+grp+'_baseMean125-binary-'+str(fold)+'-test-clean.csv', index_col = False)
    test_sequence = test['protospacer']
    test_sequence_onehot = preprocess_seq(test_sequence)
    test_label = test['significance'].to_numpy(dtype = np.float32)
    # test_annotation = test.iloc[:,np.r_[13,16:23,40,44:49]].to_numpy(dtype = np.float32)
    test_annotation = test.loc[:,feas_sel].to_numpy(dtype = np.float32)

    subsample = np.random.choice(len(test_sequence), size = 4000, replace = False)
    #test_X_sub = torch.tensor(test_sequence_onehot[subsample,:], dtype=torch.float32).to(device)
    test_X1_sub = torch.tensor(test_sequence_onehot[subsample,:], dtype=torch.float32).to(device)
    test_X2_sub = torch.tensor(test_annotation[subsample,:], dtype=torch.float32).to(device)

    dim_fc = 114

    del test_X1_sub, test_X2_sub
    test_X1 = torch.tensor(test_sequence_onehot, dtype=torch.float32).to(device)
    test_X2 = torch.tensor(test_annotation, dtype=torch.float32).to(device)
    CNN.eval()
    test_predict = sigmoid(CNN(test_X1, test_X2))
    test_predict_np = test_predict.detach().to('cpu').numpy()
    roc_auc_score(test_label, test_predict_np)
    PD = pd.DataFrame(np.stack((test['protospacer'], test_label, test_predict_np[:,0]), axis=1), columns = ['grna', 'true', 'predict'])
    PD.to_csv(resultdir + '/gRNA_binary-'+grp+'-BCE-seq-topannot-fold'+str(fold)+'-Nov28.csv')








