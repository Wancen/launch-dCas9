import torch
import torchvision
import torchvision.transforms as transforms
import itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

class DeepSeqCNN(nn.Module):
    def __init__(self, dim_y):
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
        dim_x_concat = 80
        dim_x4 = 20  # Update this based on the actual dimension of x4
        self.dim_fc_seq = dim_x_concat + dim_x4
        self.dim_fc_seq_anno = self.dim_fc_seq + dim_y
        
        self.fc2_seq = nn.Sequential(
            nn.Linear(self.dim_fc_seq, 80),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(80, 60),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(60, 40),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(40, 1),
        )
        
        self.fc2_seq_anno = nn.Sequential(
            nn.Linear(111, 80),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(80, 60),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(60, 40),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(40, 1),
        )
        
    def forward(self, x, y=None, variant="seq_anno"):
        x0 = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x).view(-1, 20)
        x_concat = torch.cat((x0, x1, x2, x3), dim=1) # size: [:,260,10]
        x_concat = x_concat.view(-1, 260*10)
        x_concat = self.fc1(x_concat)
        
        if variant == "seq":
            xy_concat = torch.cat((x_concat, x4), dim=1)
            xy_concat = self.fc2_seq(xy_concat)
        elif variant == "seq_anno":
            if y is None:
                raise ValueError("y must be provided when variant is 'seq_anno'")
            xy_concat = torch.cat((x_concat, x4, y), dim=1)
            xy_concat = self.fc2_seq_anno(xy_concat)
        else:
            raise ValueError(f"Invalid variant: {variant}")
        return xy_concat

def train_model(model, dataloader, variant, params, weight):
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    device = params['device']
    lossfunc = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([weight])).to(device)
    sigmoid = nn.Sigmoid()
    for epoch in range(params['epochs']):
        running_loss = 0.0
        for i, batch in enumerate(dataloader, 0):
            if variant == "seq_anno":
                local_x1, local_x2, local_y = batch
                local_x1, local_x2, local_y = local_x1.to(device), local_x2.to(device), local_y.to(device)
                FC_pred = model(local_x1, local_x2, variant = variant)
            else:
                local_x1, local_y = batch
                local_x1, local_y = local_x1.to(device), local_y.to(device)
                FC_pred = model(local_x1, variant = variant)
            
            optimizer.zero_grad()
            loss = lossfunc(FC_pred, local_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
                    
            if i % 200 == 199:    # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
    return model