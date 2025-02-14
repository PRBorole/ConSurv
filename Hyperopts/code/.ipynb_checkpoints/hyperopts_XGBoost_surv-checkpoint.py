import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_curve, roc_curve, auc
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

import time 
import sys 
import os
import json
import pickle
import argparse
import lifelines

import xgboost as xgb
from xgboost import plot_tree

# Get the path of the Python script
script_dir = os.path.abspath(os.path.dirname(__file__))
# Exclude script name at the end
script_dir = os.path.split(script_dir)[0]+'/'

sys.path.append(script_dir+'/../src/')
print(script_dir)
from utils import *
from utils_xgboost import *


# Read arguments
parser = argparse.ArgumentParser(description="usage help",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--index", help="index for Hyperopts/XGBoost_surv/cofigs")
args = parser.parse_args()
config = vars(args)
index = str(args.index)

with open(script_dir+'/XGBoost_surv/configs/config_'+index+'.json','r') as f:
    config = json.load(f)
f.close()

# Dataset
data_df = pd.read_csv(script_dir+'/../Data/1000_features_survival_3classes.csv',
                      index_col=0).drop(['index'],axis=1)

data_df_event_time = data_df[['event', 'time']]


data_df = pd.get_dummies(data_df.drop(['event', 'time'], axis=1),dtype=int)
scaler = MinMaxScaler()
data_df = pd.DataFrame(scaler.fit_transform(data_df), columns=data_df.columns)
data_df['event'] = [int(e) for e in data_df_event_time['event']]
data_df['time'] = data_df_event_time['time']

data_df = data_df.fillna(data_df.mean())


# split data into train and test sets
seeds = [999, 7, 42]
test_size = 0.3


result = {'seed': seeds,
          'max_depth': [config['max_depth']]*len(seeds),
          'learning_rate': [config['lr']]*len(seeds),
          'n_estimators': [config['n_estimators']]*len(seeds),
          'lambda': [config['lambda']]*len(seeds),
          'alpha': [config['alpha']]*len(seeds),
          'aft_loss_distribution': [config['aft_loss_distribution']]*len(seeds),
          'time': [],
          'c_index_train':[],
          'c_index_valid':[],
          'c_index_test':[],
          'ntrees': []}

for idx, seed in enumerate(seeds):
    data_train, data_tmp = train_test_split(data_df, test_size=test_size, random_state=seed)
    data_val, data_test = train_test_split(data_tmp, test_size=test_size, random_state=seed)
    
    X_train = data_train.drop(['event', 'time','y'], axis=1)
    # X_train = pd.get_dummies(X_train, dtype=int)
    y_lower_train = data_train['time']
    y_upper_train = np.array([t if e else np.inf for t,e in zip(data_train['time'], data_train['event'])])
    dtrain = xgb.DMatrix(X_train.values)
    dtrain.set_float_info('label_lower_bound', y_lower_train)
    dtrain.set_float_info('label_upper_bound', y_upper_train)
    
    X_val = data_val.drop(['event', 'time','y'], axis=1)
    # X_val = pd.get_dummies(X_val, dtype=int)
    y_lower_val = data_val['time']
    y_upper_val = np.array([t if e else np.inf for t,e in zip(data_val['time'], data_val['event'])])
    dvalid = xgb.DMatrix(X_val.values)
    dvalid.set_float_info('label_lower_bound', y_lower_val)
    dvalid.set_float_info('label_upper_bound', y_upper_val)
    
    X_test = data_test.drop(['event', 'time','y'], axis=1)
    # X_test = pd.get_dummies(X_test, dtype=int)
    y_lower_test = data_test['time']
    y_upper_test = np.array([t if e else np.inf for t,e in zip(data_test['time'], data_test['event'])])
    dtest = xgb.DMatrix(X_test.values)
    dtest.set_float_info('label_lower_bound', y_lower_test)
    dtest.set_float_info('label_upper_bound', y_upper_test)
    
    # Train gradient boosted trees using AFT loss and metric
    params = {'verbosity': 1,
              'objective': 'survival:aft',
              'eval_metric': 'aft-nloglik',
              'tree_method': 'hist',
              'learning_rate':config['lr'],
              'aft_loss_distribution': 'logistic',
              'aft_loss_distribution_scale': config['aft_loss_distribution'],
              'max_depth': config['max_depth'],
              'lambda': config['lambda'],
              'alpha': config['alpha']}
    
    num_boost_round = config['n_estimators']
    
    start = time.time()
    bst = xgb.train(params, dtrain, num_boost_round=num_boost_round,
                    evals=[(dtrain, 'train'), (dvalid, 'valid')],
                    early_stopping_rounds=50)
    end = time.time()
    elapsed_time = end-start
    result['time'] = result['time'] + [elapsed_time]
    
    ####################### Run prediction on the train set
    df = pd.DataFrame({'Label (lower bound)': y_lower_train,
                       'Label (upper bound)': y_upper_train,
                       'Predicted label': bst.predict(dtrain)})

    result['c_index_train'] = result['c_index_train'] + [lifelines.utils.concordance_index(event_times = data_train['time'], 
                                                                                          predicted_scores = df['Predicted label'], 
                                                                                          event_observed = data_train['event'])]

    ####################### Run prediction on the valid set
    df = pd.DataFrame({'Label (lower bound)': y_lower_val,
                       'Label (upper bound)': y_upper_val,
                       'Predicted label': bst.predict(dvalid)})

    result['c_index_valid'] = result['c_index_valid'] + [lifelines.utils.concordance_index(event_times = data_val['time'], 
                                                                                          predicted_scores = df['Predicted label'], 
                                                                                          event_observed = data_val['event'])]

    ####################### Run prediction on the test set
    df = pd.DataFrame({'Label (lower bound)': y_lower_test,
                       'Label (upper bound)': y_upper_test,
                       'Predicted label': bst.predict(dtest)})

    result['c_index_test'] = result['c_index_test'] + [lifelines.utils.concordance_index(event_times = data_test['time'], 
                                                                                        predicted_scores = df['Predicted label'], 
                                                                                        event_observed = data_test['event'])]

    result['ntrees'] = result['ntrees'] + [len(bst.get_dump())]

#### Save final results
print(result)

with open(script_dir+'/XGBoost_surv/results/result_config_'+index+'.pkl','wb') as f:
    pickle.dump(result, f)
f.close()