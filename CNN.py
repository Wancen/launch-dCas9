import torch
import argparse
import os, sys
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
import shap


parser = argparse.ArgumentParser(description='gRNA prediction model')
#parser.add_argument('--savedir', default='./', help='path to save results')
#parser.add_argument('--ckptdir', default='./ckpt', help='path to save checkpoints')
#parser.add_argument('--batch-size', type=int, default=128,
#                    help='input batch size for training (default: 128)')
# parser.add_argument('--epochs', type=int, default=60,
#                     help='number of epochs to train (default: 100)')
#parser.add_argument('--lr', type=float, default=0.001,
#                    help='learning rate (default: 0.001)')
parser.add_argument('--fold', type=int, default=1, help='which fold of data to use')
parser.add_argument('--grp', default="pro", help='promoter or enhancer region')
args = parser.parse_args()
#savedir = args.savedir
#ckptdir = args.ckptdir


#batch_size = args.batch_size
#epochs = args.epochs
#lr = args.lr
#ngpu=1

datadir = '/proj/yunligrp/users/tianyou/gRNA/data/data_fivefold/'
resultdir = '/proj/yunligrp/users/tianyou/gRNA/result/binary_fivefold/'
batch_size = 256
lr = 0.0001
ngpu=1
fold = args.fold
grp = args.grp
if grp == "enh":
    epochs = 15
elif grp == "pro":
    epochs = 60
else:
    print("Invalid group: " + grp)

print("Group: "+grp+"; Fold: "+str(fold))
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)


def preprocess_seq(data):
    print("Start preprocessing the sequence")
    length = len(data[0])
    DATA_X = np.zeros((len(data),4,length), dtype=int)
    print(DATA_X.shape)
    for l in range(len(data)):
        if l % 10000 == 0:
            print(str(l) + " sequences processed")
        for i in range(length):
            try: data[l][i]
            except: print(data[l], i, length, len(data))
            if data[l][i]in "Aa":
                DATA_X[l, 0, i] = 1
            elif data[l][i] in "Cc":
                DATA_X[l, 1, i] = 1
            elif data[l][i] in "Gg":
                DATA_X[l, 2, i] = 1
            elif data[l][i] in "Tt":
                DATA_X[l, 3, i] = 1
            else:
                print("Non-ATGC character " + data[i])
                sys.exit()
    print("Preprocessing the sequence done")
    return DATA_X



dat = pd.read_csv(datadir+'wgCERES-gRNAs-k562-discovery-screen-'+grp+'_baseMean125-binary-'+str(fold)+'-train-clean.csv', index_col = False)
sequence = dat['protospacer']
sequence_onehot = preprocess_seq(sequence)
label = dat['significance'].to_numpy(dtype = np.float32)
class_count = dat['significance'].value_counts()
w = class_count[0] / class_count[1]
'''Top features that we keep: deltagb, deltagh, H3K27ac, ATAC, DNAse, H3K4me3, TF_GATA2, OGEE_prop_Essential'''
feas_sel = ["deltagb", "deltagh", "GCcount", "GCprop", "Acount", "Ccount", "Tcount", "Gcount", "OGEE_prop_Essential", "H3k27ac_CPM_1Kb_new", 
            "DNAse_CPM_1Kb_new", "ATAC_CPM_1Kb_new", "H3K4me3_CPM_1Kb_new", "TF_GATA2_CPM_1Kb_new"]
# annotation = dat.iloc[:,np.r_[13,16:23,40,44:49]].to_numpy(dtype = np.float32)
annotation = dat.loc[:,feas_sel].to_numpy(dtype = np.float32)

X1 = torch.tensor(sequence_onehot, dtype=torch.float32)
#Xloader = torch.utils.data.DataLoader(X, batch_size=batch_size, shuffle=True)
X2 = torch.tensor(annotation, dtype=torch.float32)
Y = torch.tensor(label, dtype=torch.float32)
Y = Y.view(-1, 1)
#Yloader = torch.utils.data.DataLoader(Y, batch_size=batch_size, shuffle=True)
input_dat = TensorDataset(X1,X2,Y)
datloader = DataLoader(input_dat, batch_size=batch_size, shuffle=True)


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

class DeepSeqCNN(nn.Module):
    def __init__(self):
        super(DeepSeqCNN, self).__init__()
        # self.conv0 = nn.Sequential(
        #     nn.Conv1d(4, 50, 1),
        #     nn.MaxPool1d(2),
        #     nn.ReLU(),
        #     nn.Dropout(0.4),   ## for ReLU, it is interchangeable with max pooling and dropout
        # )
        self.conv0 = nn.Sequential(
            nn.Conv1d(4, 50, 2, stride=2),
            nn.ReLU(),
            nn.Dropout(0.4),   ## for ReLU, it is interchangeable with max pooling and dropout
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(4, 100, 3, padding=1),
            nn.MaxPool1d(2),
            nn.ReLU(),
            nn.Dropout(0.4),   ## for ReLU, it is interchangeable with max pooling and dropout
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(4, 70, 5, padding=2),
            nn.MaxPool1d(2),
            nn.ReLU(),
            nn.Dropout(0.4),   ## for ReLU, it is interchangeable with max pooling and dropout
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(4, 40, 7, padding=3),
            nn.MaxPool1d(2),  ## Their codes seemed to use average pooling but texts suggest max pooling?
            nn.ReLU(),
            nn.Dropout(0.4),   ## for ReLU, it is interchangeable with max pooling and dropout
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(4, 1, 1),
            nn.ReLU(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(260*10, 80),
            nn.ReLU(),
            nn.Dropout(0.3))
        self.fc2 = nn.Sequential(
            nn.Linear(dim_fc, 80),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(80, 60),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(60, 40),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(40, 1),
            # nn.Sigmoid()  ## BCEWithLogitsLoss takes in logits directly without sigmoid
        )
        
    def forward(self, x, y):
        x0 = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x).view(-1, 20)
        x_concat = torch.cat((x0, x1, x2, x3), dim=1) # size: [:,260,10]
        x_concat = x_concat.view(-1, 260*10)
        x_concat = self.fc1(x_concat)
        xy_concat = torch.cat((x_concat, x4, y), dim = 1)
        xy_concat = self.fc2(xy_concat)
        #for layer in self.fc:
        #    x_concat = layer(x_concat)
        #    print(x_concat.size())
        #x_concat = self.fc(x_concat)
        return xy_concat


CNN = DeepSeqCNN().to(device)
optimizer = optim.Adam(CNN.parameters(), lr=lr)
#lossfunc = nn.L1Loss().to(device)
lossfunc = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([w])).to(device)
sigmoid = nn.Sigmoid()

def train_model(model, num_epochs):
    for epoch in range(num_epochs):
        # Training
        if epoch % 2 == 0:
            model.eval()
            test_predict = sigmoid(model(test_X1_sub, test_X2_sub))
            test_predict_np = test_predict.detach().to('cpu').numpy()
            auc = roc_auc_score(test_label[subsample], test_predict_np)
            print('Epoch [%d] AUC: %.3f' %
                    (epoch + 1, auc))
            model.train()
        running_loss = 0.0
        for i, batch in enumerate(datloader, 0):
            # Transfer to GPU
            local_x1, local_x2, local_y = batch
            local_x1, local_x2, local_y = local_x1.to(device), local_x2.to(device), local_y.to(device)
            optimizer.zero_grad()
            FC_pred = model(local_x1, local_x2)
            loss = lossfunc(FC_pred, local_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 200 == 199:    # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
    
    return model

CNN = train_model(CNN, num_epochs=epochs)

ckptPATH = resultdir + '/models/gRNA_binary-'+grp+'-BCE-seq-topannot-fold'+str(fold)+'-Nov28.pth'
torch.save(CNN.state_dict(), ckptPATH)

del test_X1_sub, test_X2_sub
test_X1 = torch.tensor(test_sequence_onehot, dtype=torch.float32).to(device)
test_X2 = torch.tensor(test_annotation, dtype=torch.float32).to(device)
CNN.eval()
test_predict = sigmoid(CNN(test_X1, test_X2))
test_predict_np = test_predict.detach().to('cpu').numpy()
roc_auc_score(test_label, test_predict_np)
PD = pd.DataFrame(np.stack((test['protospacer'], test_label, test_predict_np[:,0]), axis=1), columns = ['grna', 'true', 'predict'])
PD.to_csv(resultdir + '/gRNA_binary-'+grp+'-BCE-seq-topannot-fold'+str(fold)+'-Nov28.csv')


