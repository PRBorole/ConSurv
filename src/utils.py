import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_curve, roc_curve, auc
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.metrics import matthews_corrcoef

from sklearn.preprocessing import LabelBinarizer

def get_metrics(y_true, y_pred, y_proba, y_onehot_true):

    # evaluate predictions
    accuracy = accuracy_score(y_true, y_pred)
    
    ###  MCC
    MCC = matthews_corrcoef(y_true, y_pred)
    
    ### f1 score
    f1 = f1_score(y_true, y_pred, average=None)

    ### ROC AUC
    if y_proba.shape[1]>2:
        auroc = roc_auc_score(y_true,y_proba,multi_class="ovr", average=None)
    elif y_proba.shape[1]==2:
        auroc = roc_auc_score(y_true,y_proba[:,1])
    else:
        auroc = roc_auc_score(y_true,y_proba)
    
    ### PR AUC 
    auprc = []
    for i in range(y_onehot_true.shape[1]):
        precision, recall, _ = precision_recall_curve(y_onehot_true[:,i],y_proba[:,i])
        auprc_score = auc(recall, precision)
        auprc.extend([auprc_score])

    return accuracy, MCC, f1, auroc, auprc

