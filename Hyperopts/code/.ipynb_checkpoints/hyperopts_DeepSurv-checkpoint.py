# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import os
import json
import pickle
import argparse

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

# Get the path of the Python script
script_dir = os.path.abspath(os.path.dirname(__file__))
# Exclude script name at the end
script_dir = os.path.split(script_dir)[0]+'/'

sys.path.append(script_dir+'/../src/')
print(script_dir)
from utils import *
from utils_deepsurv import *


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset


# Read arguments
parser = argparse.ArgumentParser(description="usage help",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--index", help="index for hyperopts/DeepSurv/configs")
args = parser.parse_args()
config = vars(args)
index = str(args.index)

with open(script_dir+'/DeepSurv/configs/config_'+index+'.json','r') as f:
    config = json.load(f)
f.close()

# split data into train and test sets
seeds = [999, 7, 42]
test_size = 0.3
hidden_size = config['hidden_size']  # Number of neurons in the hidden layers
l2_reg = config['l2_reg']
lr = config['lr']
batch_size = config['batch_size']
max_epochs = 250
dropout = config['dropout']

# dataset
data_df = pd.read_csv(script_dir+'/../Data/1000_features_survival_3classes.csv',
                      index_col=0).drop(['index','y'],axis=1)
data_df_event_time = data_df[['event', 'time']]


data_df = pd.get_dummies(data_df.drop(['event', 'time'], axis=1),dtype='int')
scaler = MinMaxScaler()
data_df = pd.DataFrame(scaler.fit_transform(data_df), columns=data_df.columns)
data_df['event'] = [int(e) for e in data_df_event_time['event']]
data_df['time'] = data_df_event_time['time']

data_df = data_df.fillna(data_df.mean())


result = {'seed': seeds,
          'learning_rate': [lr]*len(seeds),
          'hidden_size': [hidden_size]*len(seeds),
          'l2_reg': [l2_reg]*len(seeds),
          'batch_size': [batch_size]*len(seeds),
          'dropout': [dropout]*len(seeds),
          'time': [],
          'c_index_train':[],
          'c_index_valid':[],
          'c_index_test':[],
          'nepochs':[]}

torch.manual_seed(0)
for seed in seeds:
    
    # split data into train and test sets
    batch_size = config['batch_size']
    
    data_train, data_tmp = train_test_split(data_df, test_size=test_size, random_state=seed)
    data_val, data_test = train_test_split(data_tmp, test_size=0.5, random_state=seed)
    
    X_train = torch.tensor(data_train.drop(['event', 'time'], axis=1).to_numpy(), dtype=torch.float32)
    e_train = torch.tensor(data_train['event'].to_numpy(), dtype=torch.long)
    t_train = torch.tensor(data_train['time'].to_numpy(), dtype=torch.long)
    train_dataset = TensorDataset(X_train, e_train, t_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    X_val = torch.tensor(data_val.drop(['event', 'time'], axis=1).to_numpy(), dtype=torch.float32)
    e_val = torch.tensor(data_val['event'].to_numpy(), dtype=torch.long)
    t_val = torch.tensor(data_val['time'].to_numpy(), dtype=torch.long)
    val_dataset = TensorDataset(X_val, e_val, t_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    X_test = torch.tensor(data_test.drop(['event', 'time'], axis=1).to_numpy(), dtype=torch.float32)
    e_test = torch.tensor(data_test['event'].to_numpy(), dtype=torch.long)
    t_test = torch.tensor(data_test['time'].to_numpy(), dtype=torch.long)
    test_dataset = TensorDataset(X_test, e_test, t_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    # Example hyperparameters
    input_size = X_train.shape[1]  # Number of RNA expression features
    
    model = DeepSurv(input_size, hidden_size ,dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    
    # Train the model
    start = time.time()
    epoch = train(model, optimizer, train_loader, val_loader, max_epochs, l2_reg)
    result['nepochs'] = result['nepochs'] + [epoch]
    end = time.time()
    
    elapsed_time = end-start
    result['time'] = result['time'] + [elapsed_time]
    
    ############  EVALUATION  ###############
    
    model.eval()
    ## Training data
    events_ls = []
    times_ls = []
    predicted_ls = []
    
    for i, (inputs, events, times) in enumerate(train_loader):
        predicted_ls = predicted_ls + model(inputs).reshape(-1).tolist()
        events_ls = events_ls + events.tolist()
        times_ls = times_ls + times.tolist()
    
    
    result['c_index_train'] = result['c_index_train'] + [concordance_index(times_ls, [-i for i in predicted_ls], events_ls)]
    
    ## Validation data
    events_ls = []
    times_ls = []
    predicted_ls = []
    
    for i, (inputs, events, times) in enumerate(val_loader):
        predicted_ls = predicted_ls + model(inputs).reshape(-1).tolist()
        events_ls = events_ls + events.tolist()
        times_ls = times_ls + times.tolist()
    
    result['c_index_valid'] = result['c_index_valid'] + [concordance_index(times_ls, [-i for i in predicted_ls], events_ls)]
          
    ## test data
    events_ls = []
    times_ls = []
    predicted_ls = []
    
    for i, (inputs, events, times) in enumerate(test_loader):
        predicted_ls = predicted_ls + model(inputs).reshape(-1).tolist()
        events_ls = events_ls + events.tolist()
        times_ls = times_ls + times.tolist()
    
    result['c_index_test'] = result['c_index_test'] + [concordance_index(times_ls, [-i for i in predicted_ls], events_ls)]

#### Save final results
print(result)

with open(script_dir+'/DeepSurv/results/result_config_'+index+'.pkl','wb') as f:
    pickle.dump(result, f)
f.close()
