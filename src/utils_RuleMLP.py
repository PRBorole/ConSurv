# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import json
import pickle
import json

from tqdm import tqdm
import itertools

from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_curve, roc_curve, auc
from sklearn.metrics import make_scorer, roc_auc_score

from torchsurv.loss import cox
from lifelines.utils import concordance_index

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

class EarlyStopping:
    def __init__(self, patience=50, min_delta=0, verbose=False):
        """
        Early stops the training if validation loss doesn't improve after a given patience.
        
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset the counter if there is improvement
            if self.verbose:
                print(f"Validation loss improved to {val_loss:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping triggered")
                
class RuleMLP(nn.Module):
    def __init__(self, input_size, feature_groups):
        super(RuleMLP, self).__init__()
        
        self.feature_groups = feature_groups  # List of feature groups, each containing indices of features

        # Intermitten layer: Define separate linear transformations for each feature group
        self.intermitten_layers = nn.ModuleList([nn.Linear(len(group), len(group)) for group in feature_groups])

        # BatchNorm after the Intermitten layer
        self.bc1 = nn.BatchNorm1d(np.sum([len(group) for group in feature_groups]))
        
        # First layer: Define separate linear transformations for each feature group
        self.group_layers = nn.ModuleList([nn.Linear(len(group), 1) for group in feature_groups])
        
        # BatchNorm after the first layer
        self.bc2 = nn.BatchNorm1d(len(feature_groups),affine=False)
        
        # Output layer
        self.fc2 = nn.Linear(len(feature_groups), 1)
        
    def forward(self, x):
        # First layer: apply each group transformation to its respective input features
        intermitten_outputs = []
        for i, group in enumerate(self.feature_groups):
            intermitten_input = x[:, group]  # Extract only the features that belong to the group
            intermitten_output = self.intermitten_layers[i](intermitten_input)  # Apply the linear transformation
            intermitten_outputs.append(intermitten_output)

        # Concatenate the outputs of all groups
        intermitten_outputs = torch.cat(intermitten_outputs, dim=1)
        
        # Apply BatchNorm after first layer
        intermitten_outputs = self.bc1(intermitten_outputs)
        
        # group layer: apply each group transformation to its respective input features
        group_outputs = []
        start = 0
        for i, group in enumerate(self.feature_groups):
            group_input = intermitten_outputs[:, start:start+len(group)]  # Extract only the features that belong to the group
            start = start + len(group)
            group_output = self.group_layers[i](group_input)  # Apply the linear transformation
            group_outputs.append(group_output)
        
        # Concatenate the outputs of all groups
        out = torch.cat(group_outputs, dim=1)
        
        # Apply BatchNorm after group layer
        out = self.bc2(out)

        # Output layer
        out = self.fc2(out)
        
        return out

# Training function
def train(model, optimizer, train_loader, val_loader, num_epochs, l2_reg, patience=50):
    # Initialize early stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, events, times) in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            
            # loss = criterion(outputs, durations, events)
            neg_log_likelihood = cox.neg_partial_log_likelihood(outputs, events, times)
            
            # L2 regularization term (sum of squared weights)
            l2_reg_term = 0
            for param in model.parameters():
                l2_reg_term += torch.sum(param ** 2)
    
            # Add L2 regularization to the loss
            loss = neg_log_likelihood + l2_reg * l2_reg_term
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
    
        # Validation step
        val_loss = validate_model(model, val_loader, l2_reg)
        
        print(f"Validation Loss: {val_loss:.4f}")

        # Early stopping check
        early_stopping(val_loss)
        
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            return epoch
            break  # Exit training loop
    return epoch

def validate_model(model, val_loader, l2_reg, criterion=None):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for inputs, events, times in val_loader:
            outputs = model(inputs)
            if criterion==None:
                neg_log_likelihood = cox.neg_partial_log_likelihood(outputs, events, times)
            else:
                neg_log_likelihood = criterion(outputs, durations, events)
                
            # L2 regularization term (sum of squared weights)
            l2_reg_term = 0
            for param in model.parameters():
                l2_reg_term += torch.sum(param ** 2)
    
            # Add L2 regularization to the loss
            loss = neg_log_likelihood + l2_reg * l2_reg_term
            val_loss += loss.item()
    return val_loss / len(val_loader)  # Return average validation loss