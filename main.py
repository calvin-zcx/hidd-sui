import sys

# for linux env.
sys.path.insert(0, '..')
import pandas as pd
import time
from collections import defaultdict, Counter
import re
import pickle
import argparse
import csv
import numpy as np
import functools

print = functools.partial(print, flush=True)
from dataset import *
import random
from model import MLModels
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, roc_auc_score, confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    # Input
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument('--run_model', choices=['LSTM', 'LR', 'MLP', 'XGBOOST', 'LIGHTGBM'], default='LR')
    args = parser.parse_args()

    # Modifying args
    args.cuda = torch.cuda.is_available()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.random_seed >= 0:
        rseed = args.random_seed
    else:
        from datetime import datetime
        rseed = datetime.now()
    args.random_seed = rseed

    return args


# flaten series into static
# train_x, train_t, train_y = flatten_data(my_dataset, train_indices)  # (1764,713), (1764,), (1764,)
def flatten_data(mdata, data_indices, verbose=1):
    x, y = [], []
    for idx in data_indices:
        confounder, outcome = mdata[idx][0], mdata[idx][1]
        dx, sex, age = confounder[0], confounder[1], confounder[2]
        dx = np.sum(dx, axis=0)
        dx = np.where(dx > 0, 1, 0)

        x.append(np.concatenate((dx, [sex], age)))
        y.append(outcome)

    x, y = np.asarray(x), np.asarray(y)
    if verbose:
        d1 = len(dx)
        print('...dx:', x[:, :d1].shape, 'non-zero ratio:', (x[:, :d1] != 0).mean(), 'all-zero:',
              (x[:, :d1].mean(0) == 0).sum())
        print('...all:', x.shape, 'non-zero ratio:', (x != 0).mean(), 'all-zero:', (x.mean(0) == 0).sum())
    return x, y[:,0], y[:,1]


if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    print('args: ', args)
    print('random_seed: ', args.random_seed)

    with open(r'pickles/final_pats_1st_neg_triples.pkl', 'rb') as f:
        data_1st_neg = pickle.load(f)
        print('len(data_1st_neg):', len(data_1st_neg))

    with open(r'pickles/final_pats_1st_sui_triples.pkl', 'rb') as f:
        data_1st_sui = pickle.load(f)
        print('len(data_1st_sui):', len(data_1st_sui))

    with open(r'pickles/ccs2name.pkl', 'rb') as f:
        dx_name = pickle.load(f)
        print('len(dx_name):', len(dx_name))

    my_dataset = Dataset(data_1st_neg, diag_name=dx_name)
    my_dataset_aux = Dataset(data_1st_sui, diag_name=dx_name, diag_code_vocab=my_dataset.diag_code_vocab)

    n_feature = my_dataset.DIM_OF_CONFOUNDERS  # my_dataset.med_vocab_length + my_dataset.diag_vocab_length + 3
    feature_name = my_dataset.FEATURE_NAME
    print('n_feature: ', n_feature, ':')
    # print(feature_name)

    train_ratio = 0.9  # 0.5
    val_ratio = 0.1
    print('train_ratio: ', train_ratio,
          'val_ratio: ', val_ratio)

    dataset_size = len(my_dataset)
    indices = list(range(dataset_size))
    train_index = int(np.floor(train_ratio * dataset_size))
    np.random.shuffle(indices)

    train_indices, val_indices = indices[:train_index], indices[train_index:]
    print('Train data:')
    train_x, train_y, train_t2e = flatten_data(my_dataset, train_indices)
    print('Validation data:')
    val_x, val_y, val_t2e = flatten_data(my_dataset, val_indices)

    ###
    print('Auc data size: ', len(my_dataset_aux))
    aux_indices = list(range(len(my_dataset_aux)))
    aux_x, aux_y, aux_t2e = flatten_data(my_dataset_aux, aux_indices)
    #
    paras_grid = {
        'penalty': ['l2'],  # ['l1', 'l2'],
        'C': 10 ** np.arange(-2, 2, 0.2),  # 'C': [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20],
        'max_iter': [150],  # [100, 200, 500],
        'random_state': [args.random_seed],
        "class_weight": [None]  # ["balanced", None]
    }

    #

    model = MLModels(args.run_model, paras_grid).fit(train_x, train_y, val_x, val_y)

    # sample_weight = np.concatenate((np.ones_like(train_y), 2*np.ones_like(aux_y)))
    model_aux = MLModels(args.run_model, paras_grid).fit(
        np.concatenate((train_x, aux_x)),
        np.concatenate((train_y, aux_y)), val_x, val_y)

    print('...Original data results: ')
    print('......Results at specificity 0.9:')
    result_1 = model.performance_at_specificity(val_x, val_y, specificity=0.9)
    print('......Results at specificity 0.95:')
    result_2 = model.performance_at_specificity(val_x, val_y, specificity=0.95)
    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    print('...Using aux data results: ')
    print('......Results at specificity 0.9:')
    result_aux_1 = model_aux.performance_at_specificity(val_x, val_y, specificity=0.9)
    print('......Results at specificity 0.95:')
    result_aux_2 = model_aux.performance_at_specificity(val_x, val_y, specificity=0.95)

