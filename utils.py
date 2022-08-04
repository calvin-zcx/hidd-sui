from datetime import datetime
import os
import re
import json

import torch
import torch.utils.data
import torch.nn.functional as F

from dataset import *
# from supp_dataset_copy import *
# from supp_LSTM_dataset import *
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import softmax
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, roc_auc_score, \
    confusion_matrix


def check_and_mkdir(path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print('make dir:', dirname)
    else:
        print(dirname, 'exists, no change made')


def str_to_datetime(s):
    # input: e.g. '2016-10-14'
    # output: datetime.datetime(2016, 10, 14, 0, 0)
    # Other choices:
    #       pd.to_datetime('2016-10-14')  # very very slow
    #       datetime.strptime('2016-10-14', '%Y-%m-%d')   #  slow
    # ymd = list(map(int, s.split('-')))
    ymd = list(map(int, re.split(r'[-\/:.]', s)))
    assert (len(ymd) == 3) or (len(ymd) == 1)
    if len(ymd) == 3:
        assert 1 <= ymd[1] <= 12
        assert 1 <= ymd[2] <= 31
    elif len(ymd) == 1:
        ymd = ymd + [1, 1]  # If only year, set to Year-Jan.-1st
    return datetime(*ymd)


# New added 2021/05/04 Zang
def check_and_mkdir(path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print('make dir:', dirname)
    else:
        print(dirname, 'exists, no change made')


def load_model(model_class, filename):
    def _map_location(storage, loc):
        return storage

    # load trained on GPU models to CPU
    map_location = None
    if not torch.cuda.is_available():
        map_location = _map_location

    state = torch.load(str(filename), map_location=map_location)

    model = model_class(**state['model_params'])
    model.load_state_dict(state['model_state'])

    return model


def save_model(model, filename, model_params=None):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    state = {
        'model_params': model_params or {},
        'model_state': model.state_dict(),
    }
    check_and_mkdir(str(filename))
    torch.save(state, str(filename))


def logits_to_probability(logits, normalized):
    if normalized:
        if len(logits.shape) == 1:
            return logits
        elif len(logits.shape) == 2:
            return logits[:, 1]
        else:
            raise ValueError
    else:
        if len(logits.shape) == 1:
            return 1 / (1 + np.exp(-logits))
        elif len(logits.shape) == 2:
            prop = softmax(logits, axis=1)
            return prop[:, 1]
        else:
            raise ValueError


def transfer_data(model, dataloader, cuda=True, normalized=False):
    with torch.no_grad():
        model.eval()
        loss_list = []
        logits_list = []
        Y_list = []
        uid_list = []
        X_list = []
        X_embed_list = []

        for X, Y, outcome, uid in dataloader:
            if cuda:
                for i in range(len(X)):
                    X[i] = X[i].to('cuda')
                Y = Y.float().to('cuda')

            logits, _ = model(X)
            if isinstance(logits, tuple):
                logits = logits[0]

            # loss = F.cross_entropy(logits, Y)
            # loss = nn.CrossEntropyLoss()(logits, labels)

            loss = F.binary_cross_entropy_with_logits(logits, Y)

            if cuda:
                logits = logits.to('cpu').detach().data.numpy()
                Y = Y.to('cpu').detach().data.numpy()
                loss = loss.to('cpu').detach().data.numpy()
            else:
                logits = logits.detach().data.numpy()
                Y = Y.detach().data.numpy()
                loss = loss.detach().data.numpy()

            logits_list.append(logits)  # [:,1])
            Y_list.append(Y)
            loss_list.append(loss.item())
            uid_list.append(uid)

        loss_final = np.mean(loss_list)
        Y_final = np.concatenate(Y_list)
        # X_final = np.concatenate(X_list)
        # X_embed_final = np.concatenate(X_embed)
        logits_final = np.concatenate(logits_list)
        uid_final = np.concatenate(uid_list)

        Y_pred_final = logits_to_probability(logits_final, normalized=normalized)
        auc = roc_auc_score(Y_final, Y_pred_final)

        return auc, loss_final, Y_final, Y_pred_final, uid_final


def print_records_of_uid(uid, fname=r'pickles/final_pats_1st_neg_dict_before20150930.pkl'):
    # e.g. uid = '937504'
    # debug: print features of one patient:
    with open(fname, 'rb') as f:
        datadict = pickle.load(f)
        if os.path.exists('pickles/icd_des.pkl'):
            with open(r'pickles/icd_des.pkl', 'rb') as f2:
                icd_des = pickle.load(f2)
        else:
            icd = pd.read_excel('data/CMS32_DESC_LONG_SHORT_DX.xlsx ')
            icd_des = {r[0]: r[2] for i, r in icd.iterrows()}
            pickle.dump(icd_des, open('pickles/icd_des.pkl', 'wb'))
        print('len(data_1st_neg):', len(datadict))
        print('len(icd_des):', len(icd_des))

    arecord = datadict.get(uid)
    a_details = []
    for a in arecord:
        a_details.append(a[:4] + [x + '_' + icd_des.get(x, '') for x in a[4:]])
    for a in a_details:
        print(a)
    return a_details, datadict


def x_target_to_source(target_x, target_vocab, source_x_dim, source_vocab):
    target_x_transform = np.zeros((target_x.shape[0], source_x_dim))
    # direct transform non-diagnostic dimensions len(target_vocab):target_x.shape[1]
    target_x_transform[:, len(source_vocab):] = target_x[:, len(target_vocab):]
    # mapping diagnostic dimensions 0:len(target_vocab)
    for i in range(len(target_vocab)):
        code = target_vocab.id2code[i]
        col = source_vocab.code2id.get(code, -1)
        if (col >= 0) and (col < target_x_transform.shape[1]):
            target_x_transform[:, col] = target_x[:, i]
        else:
            print('{}th code {} (cnt:{} {}) from target not exist in source vocabulary'.format(
                i, code, target_vocab.code2count[code], target_vocab.id2name[i]))
    return target_x_transform
