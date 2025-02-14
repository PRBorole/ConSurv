# import libraries
import pandas as pd
import numpy as np

from rulekit.classification import RuleClassifier
from rulekit.params import Measures

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_curve, roc_curve, auc
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import LabelBinarizer

import time 
import sys 
import os
import json
import pickle
import argparse

# Get the path of the Python script
script_dir = os.path.abspath(os.path.dirname(__file__))
# Exclude script name at the end
script_dir = os.path.split(script_dir)[0]+'/'

sys.path.append(script_dir+'/../src/')
print(script_dir)
from utils import *

# Read arguments
parser = argparse.ArgumentParser(description="usage help",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--index", help="index for Hyperopts/RuleKit/cofigs")
args = parser.parse_args()
config = vars(args)
index = str(args.index)

with open(script_dir+'/configs/config_'+index+'.json','r') as f:
    config = json.load(f)
f.close()

if config['measures'] == 'c2':
    measure = Measures.C2
elif config['measures'] == 'rss':
    measure = Measures.RSS
elif config['measures'] == 'correlation':
    measure = Measures.Correlation

# Dataset
nclasses = 3
data_df = pd.read_csv(script_dir+'/../Data/1000_features_survival_3classes.csv',
                      index_col=0).drop(['index'],axis=1)

# Initialize the LabelEncoder
le = LabelEncoder()

# Loop through each column and apply encoding to object type columns
for col in data_df.columns:
    if data_df[col].dtype == 'object':
        data_df[col] = le.fit_transform(data_df[col])

data_df = data_df.fillna(data_df.mean())

X = data_df.drop(['event', 'time','y'], axis=1)
y = data_df['y']



# split data into train and test sets
seeds = [999, 7, 42]
test_size = 0.3

result = {'seed': seeds,
          'measure': [config['measures']]*len(seeds),
          'minsupp_new': [config['minsupp_new']]**len(seeds),
          'time': [None]*len(seeds),
          'accuracy_train': [None]*len(seeds),
          'MCC_train': [None]*len(seeds),
          'f1_train': [None]*len(seeds), 
          'auroc_train': [None]*len(seeds),
          'auprc_train': [None]*len(seeds),
          'accuracy_val': [None]*len(seeds),
          'MCC_val': [None]*len(seeds),
          'f1_val': [None]*len(seeds), 
          'auroc_val': [None]*len(seeds),
          'auprc_val': [None]*len(seeds),
          'accuracy_test': [None]*len(seeds),
          'MCC_test': [None]*len(seeds),
          'f1_test': [None]*len(seeds), 
          'auroc_test': [None]*len(seeds),
          'auprc_test': [None]*len(seeds),
          'nrules': [None]*len(seeds),
          'rules_count': [None]*len(seeds),
          'conditions_per_rule': [None]*len(seeds),
          'induced_conditions_per_rule': [None]*len(seeds),
          'avg_rule_coverage': [None]*len(seeds),
          'avg_rule_precision': [None]*len(seeds),
          'avg_rule_quality': [None]*len(seeds),
          'pvalue': [None]*len(seeds)}

for idx, seed in enumerate(seeds):
    # split data into train and test sets
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)

    
    classifier = RuleClassifier(
        induction_measure=measure,
        pruning_measure=measure,
        voting_measure=measure,
        minsupp_new=config['minsupp_new']
    )
    
    start = time.time()
    classifier.fit(X_train, y_train)
    end = time.time()
    elapsed_time = end-start
    print("Time for training: ", elapsed_time)
    result['time'][idx] = [elapsed_time]
    ruleset = classifier.model
    
    mapper_dict = {class_:idx for idx, class_ in enumerate(classifier.label_unique_values)}

    # make predictions for train data
    y_train_pred = classifier.predict(X_train)
    y_train_proba = classifier.predict_proba(X_train)
    
    ## need this correction because rulekit changes classes order randomly (maybe based on number of instances)
    y_train_proba_corrected = [None]*nclasses
    for class_ in range(nclasses):
        y_train_proba_corrected[class_] =  y_train_proba[:,mapper_dict[class_]]
    y_train_proba_corrected = np.array(y_train_proba_corrected).T
    
    # make predictions for val data
    y_val_pred = classifier.predict(X_val)
    y_val_proba = classifier.predict_proba(X_val)
    
    ## need this correction because rulekit changes classes order randomly (maybe based on number of instances)
    y_val_proba_corrected = [None]*nclasses
    for class_ in range(nclasses):
        y_val_proba_corrected[class_] =  y_val_proba[:,mapper_dict[class_]]
    y_val_proba_corrected = np.array(y_val_proba_corrected).T
    
    # make predictions for test data
    y_test_pred = classifier.predict(X_test)
    y_test_proba = classifier.predict_proba(X_test)
    
    ## need this correction because rulekit changes classes order randomly (maybe based on number of instances)
    y_test_proba_corrected = [None]*nclasses
    for class_ in range(nclasses):
        y_test_proba_corrected[class_] =  y_test_proba[:,mapper_dict[class_]]
    y_test_proba_corrected = np.array(y_test_proba_corrected).T
    
    # make binary labels
    label_binarizer = LabelBinarizer().fit(y_train)

    ################### RESULTS
    ## Train
    print("*********************** TRAIN ***********************")
    y_onehot_train = label_binarizer.transform(y_train)
    accuracy, MCC, f1, auroc, auprc = get_metrics(y_train, y_train_pred, y_train_proba_corrected, y_onehot_train)
    print("accuracy: ", accuracy)
    print("MCC: ", MCC)
    print("f1: ", f1)
    print("auroc: ", auroc)
    print("auprc: ", auprc)
    result['accuracy_train'][idx] = accuracy
    result['MCC_train'][idx] = MCC
    result['f1_train'][idx] = f1
    result['auroc_train'][idx] = auroc
    result['auprc_train'][idx] = auprc
        
    ## valid
    y_onehot_val = label_binarizer.transform(y_val)
    print("*********************** VALID ***********************")
    accuracy, MCC, f1, auroc, auprc = get_metrics(y_val, y_val_pred, y_val_proba_corrected, y_onehot_val)
    print("accuracy: ", accuracy)
    print("MCC: ", MCC)
    print("f1: ", f1)
    print("auroc: ", auroc)
    print("auprc: ", auprc)
    result['accuracy_val'][idx] = accuracy
    result['MCC_val'][idx] = MCC
    result['f1_val'][idx] = f1
    result['auroc_val'][idx] = auroc
    result['auprc_val'][idx] = auprc
    
    ## test
    y_onehot_test = label_binarizer.transform(y_test)
    print("*********************** TEST ***********************")
    accuracy, MCC, f1, auroc, auprc = get_metrics(y_test, y_test_pred, y_test_proba_corrected, y_onehot_test)
    print("accuracy: ", accuracy)
    print("MCC: ", MCC)
    print("f1: ", f1)
    print("auroc: ", auroc)
    print("auprc: ", auprc)
    result['accuracy_test'][idx] = accuracy
    result['MCC_test'][idx] = MCC
    result['f1_test'][idx] = f1
    result['auroc_test'][idx] = auroc
    result['auprc_test'][idx] = auprc
    
    print("Number of rules: ", len(ruleset.rules))
    result['nrules'][idx] = len(ruleset.rules)
    
    # Get the stats
    tmp_dict = vars(classifier.model.stats)
    result['rules_count'][idx] = tmp_dict['rules_count']
    result['conditions_per_rule'][idx] = tmp_dict['conditions_per_rule']
    result['induced_conditions_per_rule'][idx] = tmp_dict['induced_conditions_per_rule']
    result['avg_rule_coverage'][idx] = tmp_dict['avg_rule_coverage']
    result['avg_rule_precision'][idx] = tmp_dict['avg_rule_precision']
    result['avg_rule_quality'][idx] = tmp_dict['avg_rule_quality']
    result['pvalue'][idx] = tmp_dict['pvalue']


# Saving to a file (write mode)

with open(script_dir+'/RuleKit/results/result_config_'+str(index)+'.pkl','wb') as f:
    pickle.dump(result, f)
f.close()
