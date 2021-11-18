from datetime import datetime
import os
# import pandas as pd
# import time
# import pickle
# import csv
# import numpy as np
import re
import json

import torch
import torch.utils.data
from dataset import *
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from collections import Counter, defaultdict
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


def transfer_data(model, dataloader, cuda=True, normalized=False,
                  pretrain_model=None, exclude_1_visit=False):
    with torch.no_grad():
        model.eval()
        loss_list = []
        logits_list = []
        Y_list = []
        uid_list = []
        X_list = []
        X_embed_list = []

        for X, Y, Y_t2e, Y_more, idx in dataloader:
            if cuda:
                X = X.float().to('cuda')
                Y = Y.long().to('cuda')
                Y_more = Y_more.long().to('cuda')

            if pretrain_model:
                with torch.no_grad():
                    pretrain_model.eval()
                    X_embed = pretrain_model.encoder(X)
            else:
                X_embed = X
            if not exclude_1_visit:
                Y[:, 0] = Y[:, 0] + Y[:, 2]
            _, labels = Y[:, :2].max(dim=1)
            Y = labels
            logits = model(X_embed)
            if isinstance(logits, tuple):
                logits = logits[0]
            # loss = F.cross_entropy(logits, Y)
            loss = nn.CrossEntropyLoss()(logits, labels)

            if cuda:
                logits = logits.to('cpu').detach().data.numpy()
                Y = Y.to('cpu').detach().data.numpy()
                X = X.to('cpu').detach().data.numpy()
                X_embed = X_embed.to('cpu').detach().data.numpy()
                loss = loss.to('cpu').detach().data.numpy()
            else:
                logits = logits.detach().data.numpy()
                Y = Y.detach().data.numpy()
                X = X.detach().data.numpy()
                X_embed = X_embed.detach().data.numpy()
                loss = loss.detach().data.numpy()

            logits_list.append(logits)  # [:,1])
            Y_list.append(Y)
            X_list.append(X)
            X_embed_list.append(X_embed)
            loss_list.append(loss.item())
            uid_list.append(idx)

        loss_final = np.mean(loss_list)
        Y_final = np.concatenate(Y_list)
        X_final = np.concatenate(X_list)
        X_embed_final = np.concatenate(X_embed)
        logits_final = np.concatenate(logits_list)
        uid_final = np.concatenate(uid_list)

        Y_pred_final = logits_to_probability(logits_final, normalized=normalized)
        auc = roc_auc_score(Y_final, Y_pred_final)

        return auc, loss_final, Y_final, Y_pred_final, uid_final, X_final, X_embed_final


def tsne_plot(x, y, perplexity = 50, dump=False, fname='tsne'):
    emethod = 'tsne'
    # x, y, t2e, all_uid = flatten_data(my_dataset, indices)
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, init='pca')  # , n_iter=300)
    results = tsne.fit_transform(x)
    # df = pd.DataFrame(data={'y': y.astype(int),
    #                         '{}-Dim-1'.format(emethod): results[:, 0],
    #                         '{}-Dim-2'.format(emethod): results[:, 1]})
    markers = ['o', 's', 'p', 'x', '^', '+', '*', '<', 'D', 'h', '>']
    # markers = ['o', 'x']
    colors = ['#F65453', '#82A2D3', '#FAC200', 'purple']

    # plt.figure(figsize=(16, 10))
    # fig, ax = plt.subplots(figsize=(12, 8))
    plt.figure(figsize=(12, 8))

    plt.scatter(results[np.where(y[:, 2] == 1), 0], results[np.where(y[:, 2] == 1), 1], marker='s', c='b',
                  alpha=0.5, label='One visit')
    plt.scatter(results[np.where(y[:, 3] == 1), 0], results[np.where(y[:, 3] == 1), 1], marker='+', c='purple',
                alpha=0.5, label='1st suicide')
    plt.scatter(results[(y[:,0] == 1), 0], results[(y[:,0] == 1), 1], marker='o', c='g', alpha=0.5, label='Negative')
    plt.scatter(results[np.where(y[:, 1] == 1), 0], results[np.where(y[:, 1] == 1), 1], s=100, marker='x', c='r',
                alpha=1, label='Positive')
    plt.legend(loc='best')
    # ax = sns.scatterplot(
    #     x="{}-Dim-1".format(emethod),
    #     y="{}-Dim-2".format(emethod),
    #     hue="y",
    #     # palette=sns.color_palette("hls", 2),
    #     data=df,
    #     legend="full",
    #     alpha=0.5,
    #     style="y")  # , s=30)

    # fig = ax.get_figure()
    if dump:
        plt.savefig('figure/{}-per{}.png'.format(fname, perplexity))
        plt.savefig('figure/{}-per{}.pdf'.format(fname, perplexity))
    plt.show()
    plt.close()


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
