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
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, roc_auc_score, \
    confusion_matrix
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


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
def flatten_data(mdata, data_indices, verbose=1, bool=True):
    x, y = [], []
    uid_list = []
    for idx in data_indices:
        confounder, outcome, uid = mdata[idx][0], mdata[idx][1], mdata[idx][2]
        dx, sex, age = confounder[0], confounder[1], confounder[2]
        # if uid in ['2042577', '1169413']:
        #     print(uid)
        dx = np.sum(dx, axis=0)
        if bool:
            dx = np.where(dx > 0, 1, 0)

        x.append(np.concatenate((dx, [sex], age)))
        y.append(outcome)
        uid_list.append(uid)

    x, y = np.asarray(x), np.asarray(y)
    if verbose:
        d1 = len(dx)
        print('...dx:', x[:, :d1].shape, 'non-zero ratio:', (x[:, :d1] != 0).mean(), 'all-zero:',
              (x[:, :d1].mean(0) == 0).sum())
        print('...all:', x.shape, 'non-zero ratio:', (x != 0).mean(), 'all-zero:', (x.mean(0) == 0).sum())
    return x, y[:, 0], y[:, 1], uid_list


if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    print('args: ', args)
    print('random_seed: ', args.random_seed)

    with open(r'pickles/final_pats_1st_neg_triples_exclude1visit.pkl', 'rb') as f:
        # final_pats_1st_neg_triples_exclude1visit.pkl
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

    train_ratio = 0.8  # 0.5
    val_ratio = 0.1
    print('train_ratio: ', train_ratio,
          'val_ratio: ', val_ratio,
          'test_ratio: ', 1 - (train_ratio + val_ratio))

    dataset_size = len(my_dataset)
    indices = list(range(dataset_size))
    train_index = int(np.floor(train_ratio * dataset_size))
    val_index = int(np.floor(val_ratio * dataset_size))
    np.random.shuffle(indices)

    train_indices, val_indices, test_indices = indices[:train_index], \
                                               indices[train_index:train_index + val_index], \
                                               indices[train_index + val_index:]

    print('Train data:')
    train_x, train_y, train_t2e, train_uid = flatten_data(my_dataset, train_indices)
    print('Validation data:')
    val_x, val_y, val_t2e, val_uid = flatten_data(my_dataset, val_indices)
    print('Test data:')
    test_x, test_y, test_t2e, test_uid = flatten_data(my_dataset, test_indices)

    ###
    print('Auc data size: ', len(my_dataset_aux))
    aux_indices = list(range(len(my_dataset_aux)))
    aux_x, aux_y, aux_t2e, aux_uid = flatten_data(my_dataset_aux, aux_indices)
    idx_sui = np.where(feature_name == 'Suicide and intentional self-inflicted injury')

    if args.run_model == 'LR':
        paras_grid = {
            'penalty': ['l1', 'l2'],
            'C': 10 ** np.arange(-2, 2, 0.2),  # 'C': [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20],
            'max_iter': [150],  # [100, 200, 500],
            'random_state': [args.random_seed],
            # "class_weight": [None, "balanced"],
        }
    elif args.run_model == 'LIGHTGBM':
        paras_grid = {
            'max_depth': [3, 4, 5],
            'learning_rate': np.arange(0.01, 1, 0.1),
            'num_leaves': np.arange(5, 50, 10),
            'min_child_samples': [50, 100, 150, 200, 250, 300],
            'random_state': [args.random_seed],
        }
    #

    model = MLModels(args.run_model, paras_grid).fit(train_x, train_y, val_x, val_y)

    sample_weight = np.concatenate((np.ones_like(train_y), 1 * np.ones_like(aux_y)))
    model_aux = MLModels(args.run_model, paras_grid).fit(
        np.concatenate((train_x, aux_x)),
        np.concatenate((train_y, aux_y)), val_x, val_y, sample_weight=sample_weight)

    print('...Original data results: ')
    print('......Results at specificity 0.9:')
    result_1 = model.performance_at_specificity(test_x, test_y, specificity=0.9)
    print('......Results at specificity 0.95:')
    result_2 = model.performance_at_specificity(test_x, test_y, specificity=0.95)
    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    print('...Using aux data results: ')
    print('......Results at specificity 0.9:')
    result_aux_1 = model_aux.performance_at_specificity(test_x, test_y, specificity=0.9)
    print('......Results at specificity 0.95:')
    result_aux_2 = model_aux.performance_at_specificity(test_x, test_y, specificity=0.95)

    df1 = pd.DataFrame([result_1 + (model.best_hyper_paras,), result_2 + (model.best_hyper_paras,),
                        result_aux_1 + (model_aux.best_hyper_paras,), result_aux_2 + (model_aux.best_hyper_paras,)],
                       columns=["AUC", "threshold", "Specificity", "Sensitivity/recall", "PPV/precision",
                                "n_negative", "n_positive", "precision_recall_fscore_support", 'best_hyper_paras'],
                       index=['r_9', 'r_95', 'raux_9', 'raux_95'])
    df1.to_csv('output/test_results_{}r{}.csv'.format(args.run_model, args.random_seed))
    df2 = pd.DataFrame({'aux_x': aux_x.mean(axis=0), 'train_x': train_x.mean(axis=0),
                        'train_x_pos': train_x[train_y == 1].mean(axis=0)},
                       index=feature_name).reset_index()
    df2.to_csv('output/data_train_vs_aux_{}r{}.csv'.format(args.run_model, args.random_seed))

    if args.run_model == 'LR':
        df3 = pd.DataFrame({'aux_model': model_aux.best_model.coef_[0], 'train_x_model': model.best_model.coef_[0]},
                           index=feature_name).reset_index()
        df3.to_csv('output/model_coef_train_vs_aux_LRr{}.csv'.format(args.random_seed))

    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    test_y_pre_prob = model.predict_proba(test_x)
    auc = roc_auc_score(test_y, test_y_pre_prob)
    threshold = result_2[1]
    test_y_pre = (test_y_pre_prob > threshold).astype(int)
    r = precision_recall_fscore_support(test_y, test_y_pre)
    print('precision_recall_fscore_support:\n', r)
    feat = [';'.join(feature_name[np.nonzero(test_x[i,:])[0]]) for i in range(len(test_x))]
    pd.DataFrame({'test_uid': test_uid, 'test_y_pre_prob': test_y_pre_prob, 'test_y_pre': test_y_pre, 'test_y': test_y, 'feat':feat}).to_csv(
        'output/test_pre_details_{}r{}.csv'.format(args.run_model, args.random_seed))


    # with open(r'pickles/final_pats_1st_neg_dict.pkl', 'rb') as f:
    #     datadict = pickle.load(f)
    #     print('len(data_1st_neg):', len(datadict))

    # pickle.dump((model, model_aux), open('pickles/models.pkl', 'wb'))
    # perplexity = 50  # 35 # 25 #50 # 25 # 50
    # emethod = 'tsne'  # "tsne" #
    # x, y, t2e, all_uid = flatten_data(my_dataset, indices)
    # tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, init='pca')  # , n_iter=300)
    # results = tsne.fit_transform(x)
    #
    # df = pd.DataFrame(data={'y': y.astype(int),
    #                         '{}-Dim-1'.format(emethod): results[:, 0],
    #                         '{}-Dim-2'.format(emethod): results[:, 1]})
    # markers = ['o', 's', 'p', 'x', '^', '+', '*', '<', 'D', 'h', '>']
    # markers = ['o', 'x']
    #
    # # plt.figure(figsize=(16, 10))
    # ax = sns.scatterplot(
    #     x="{}-Dim-1".format(emethod),
    #     y="{}-Dim-2".format(emethod),
    #     hue="y",
    #     # palette=sns.color_palette("hls", 2),
    #     data=df,
    #     legend="full",
    #     alpha=0.5,
    #     style="y")  # , s=30)
    #
    # fig = ax.get_figure()
    # plt.show()
    # fig.savefig( 'figure/per{}-pcainit.png'.format(perplexity))
    # fig.savefig( 'figure/per{}-pcainit.pdf'.format(perplexity))
    #
