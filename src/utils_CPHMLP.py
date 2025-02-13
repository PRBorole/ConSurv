# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
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


class CPHMLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CPHMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bc1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bc2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.bc1(out)
        out = self.fc2(out)
        out = self.bc2(out)
        out = self.fc3(out)
        return out

# Training function
def train(model, optimizer, train_loader, val_loader, num_epochs, l2_reg):
    # Initialize early stopping object
    early_stopping = EarlyStopping(verbose=True)
    
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


# Define the neural network
# class CoxPHLoss(torch.nn.Module):
#     def __init__(self, model, l2_reg=0.001):
#         """
#         Initialize the Cox PH loss with L2 regularization.

#         Args:
#             model (torch.nn.Module): The neural network model.
#             l2_reg (float): The L2 regularization strength (lambda).
#         """
#         super(CoxPHLoss, self).__init__()
#         self.model = model  # The neural network model whose weights we want to regularize
#         self.l2_reg = l2_reg

#     def forward(self, predicted_risks, durations, events):
#         """
#         Compute the Cox Proportional Hazards loss with L2 regularization.

#         Args:
#             predicted_risks (torch.Tensor): Predicted log-risk (output of the network).
#             durations (torch.Tensor): Observed times (either event time or censoring time).
#             events (torch.Tensor): Event indicators (1 if event occurred, 0 if censored).
        
#         Returns:
#             torch.Tensor: Negative log partial likelihood loss with L2 regularization.
#         """
#         # Ensure that the tensors are the correct shape
#         predicted_risks = predicted_risks.squeeze()
#         durations = durations.squeeze()
#         events = events.squeeze()

#         # Sort by durations in descending order (longer durations first)
#         sorted_indices = torch.argsort(durations, descending=True)
#         sorted_risks = predicted_risks[sorted_indices]
#         sorted_events = events[sorted_indices]

        
#         # Compute the cumulative sum of the exponentiated risks
#         hazard_ratio = torch.exp(sorted_risks)
#         cumulative_hazard_ratio = torch.cumsum(hazard_ratio, dim=0)
        
#         # Compute log of cumulative hazard ratios
#         log_cumulative_hazard_ratio = torch.log(cumulative_hazard_ratio)

#         # Calculate the log partial likelihood only for events (E = 1)
#         uncensored_likelihood = sorted_risks - log_cumulative_hazard_ratio
#         censored_likelihood = uncensored_likelihood * sorted_events
        
#         # Compute the negative log partial likelihood
#         neg_log_likelihood = -torch.sum(censored_likelihood) / torch.sum(sorted_events)

#         # L2 regularization term (sum of squared weights)
#         l2_reg_term = 0
#         for param in self.model.parameters():
#             l2_reg_term += torch.sum(param ** 2)

#         # Add L2 regularization to the loss
#         total_loss = neg_log_likelihood + self.l2_reg * l2_reg_term
#         # print("sorted: ", sorted_risks)
#         # print("hazard_ratio: ", hazard_ratio)
#         # print("cumulative_hazard_ratio: ", cumulative_hazard_ratio)
#         # print("log_cumulative_hazard_ratio: ", log_cumulative_hazard_ratio)
#         # print("uncensored_likelihood: ", uncensored_likelihood)
#         # print("censored_likelihood: ", censored_likelihood)
#         # print("torch.sum(sorted_events): ", torch.sum(censored_likelihood))
        
#         # print("l2_reg_term: ", l2_reg_term)
#         # print("total_loss: ", total_loss)
        
#         return total_loss
