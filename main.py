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
import os
from torch.utils.data.sampler import SubsetRandomSampler
from scipy.special import softmax

print = functools.partial(print, flush=True)
from dataset import *
import random
import itertools
from model import ml, mlp
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, roc_auc_score, \
    confusion_matrix
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import torch.nn as nn
import torch.nn.functional as F
from utils import save_model, load_model, check_and_mkdir


def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    # Input
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument('--run_model', choices=['LSTM', 'LR', 'MLP', 'XGBOOST', 'LIGHTGBM'], default='LR')
    # Deep PSModels
    parser.add_argument('--batch_size', type=int, default=256)  # 768)  # 64)
    parser.add_argument('--learning_rate', type=float, default=1e-3)  # 0.001
    parser.add_argument('--weight_decay', type=float, default=1e-6)  # )0001)
    parser.add_argument('--epochs', type=int, default=20)  # 30
    # MLP
    parser.add_argument('--hidden_size', type=str, default='', help=', delimited integers')
    # Output
    parser.add_argument('--output_dir', type=str, default='output/')
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

    args.save_model_filename = os.path.join(args.output_dir, 'S{}_{}'.format(args.random_seed, args.run_model))
    check_and_mkdir(args.save_model_filename)

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

        for confounder, outcome, uid in dataloader:
            dx, sex, age = confounder[0], confounder[1], confounder[2]
            dx = torch.sum(dx, 1)
            dx = torch.where(dx > 0, 1., 0.)
            X = torch.cat((dx, sex.unsqueeze(1), age), 1)
            Y = outcome[:, 0]
            if cuda:
                X = X.float().to('cuda')
                Y = Y.long().to('cuda')

            logits = model(X)
            loss = F.cross_entropy(logits, Y)

            if cuda:
                logits = logits.to('cpu').detach().data.numpy()
                Y = Y.to('cpu').detach().data.numpy()
                X = X.to('cpu').detach().data.numpy()
                loss = loss.to('cpu').detach().data.numpy()
            else:
                logits = logits.detach().data.numpy()
                Y = Y.detach().data.numpy()
                X = X.detach().data.numpy()
                loss = loss.detach().data.numpy()

            logits_list.append(logits)  # [:,1])
            Y_list.append(Y)
            X_list.append(X)
            loss_list.append(loss.item())
            uid_list.append(uid)

        loss_final = np.mean(loss_list)
        Y_final = np.concatenate(Y_list)
        X_final = np.concatenate(X_list)
        logits_final = np.concatenate(logits_list)
        uid_final = np.concatenate(uid_list)

        Y_pred_final = logits_to_probability(logits_final, normalized=normalized)
        auc = roc_auc_score(Y_final, Y_pred_final)

        return auc, loss_final, Y_final, Y_pred_final, uid_final, X_final


if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    print('args: ', args)
    print('random_seed: ', args.random_seed)

    with open(r'pickles/final_pats_1st_neg_triples_before20150930.pkl', 'rb') as f:
        # final_pats_1st_neg_triples_exclude1visit.pkl
        data_1st_neg = pickle.load(f)
        print('len(data_1st_neg):', len(data_1st_neg))

    with open(r'pickles/final_pats_1st_sui_triples_before20150930.pkl', 'rb') as f:
        data_1st_sui = pickle.load(f)
        print('len(data_1st_sui):', len(data_1st_sui))

    with open(r'pickles/ccs2name.pkl', 'rb') as f:
        dx_name = pickle.load(f)
        print('len(dx_name):', len(dx_name))

    my_dataset = Dataset(data_1st_neg, diag_name=dx_name)
    ttttt = my_dataset._to_tensor()
    my_dataset_aux = Dataset(data_1st_sui, diag_name=dx_name, diag_code_vocab=my_dataset.diag_code_vocab)

    n_feature = my_dataset.DIM_OF_CONFOUNDERS  # my_dataset.med_vocab_length + my_dataset.diag_vocab_length + 3
    feature_name = my_dataset.FEATURE_NAME
    print('n_feature: ', n_feature, ':')
    # print(feature_name)

    # change later: 0.7:0.1:0.2 to see the difference
    train_ratio = 0.7  # 0.8  # 0.5
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
    if args.run_model in ['MLP', 'AE']:
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_loader = torch.utils.data.DataLoader(my_dataset, batch_size=args.batch_size,
                                                   sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(my_dataset, batch_size=args.batch_size,
                                                 sampler=val_sampler)
        test_loader = torch.utils.data.DataLoader(my_dataset, batch_size=args.batch_size,
                                                  sampler=test_sampler)
        data_loader = torch.utils.data.DataLoader(my_dataset, batch_size=args.batch_size,
                                                  sampler=SubsetRandomSampler(indices))
        if args.run_model == 'MLP':
            print("**************************************************")
            print("**************************************************")
            print(args.run_model, ' model learning:')

            # PSModels configuration & training
            # paras_grid = {
            #     'hidden_size': [0, 32, 64, 128],
            #     'lr': [1e-2, 1e-3, 1e-4],
            #     'weight_decay': [1e-4, 1e-5, 1e-6],
            #     'batch_size': [32, 64, 128],
            #     'dropout': [0.5],
            # }
            paras_grid = {
                'hidden_size': [32, 64, [32,32], [64,64]],  #, 64, 128],
                'lr': [1e-3, 1e-4],
                'weight_decay': [1e-4, 1e-5, 1e-6],
                'batch_size': [512],
                'dropout': [0.5],
            }
            hyper_paras_names, hyper_paras_v = zip(*paras_grid.items())
            hyper_paras_list = list(itertools.product(*hyper_paras_v))
            print('Model {} Searching Space N={}: '.format(args.run_model, len(hyper_paras_list)), paras_grid)

            best_hyper_paras = None
            best_model = None
            best_auc = float('-inf')
            best_model_epoch = -1
            results = []
            i = -1
            i_iter = -1
            for hyper_paras in tqdm(hyper_paras_list):
                i += 1
                hidden_size, lr, weight_decay, batch_size, dropout = hyper_paras
                print('In hyper-paras space [{}/{}]...'.format(i, len(hyper_paras_list)))
                print(hyper_paras_names)
                print(hyper_paras)

                train_loader_shuffled = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size,
                                                                    sampler=train_sampler)
                print('len(train_loader_shuffled): ', len(train_loader_shuffled),
                      'train_loader_shuffled.batch_size: ', train_loader_shuffled.batch_size)

                model_params = dict(input_size=n_feature, num_classes=2, hidden_size=hidden_size, dropout=dropout, )
                print('Model: MLP')
                print(model_params)
                model = mlp.MLP(**model_params)
                if args.cuda:
                    model = model.to('cuda')
                print(model)

                optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
                # optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
                # scheduler = torch.optim.lr_scheduler.StepLR( optimizer, step_size=5, gamma=0.1)
                # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.85, verbose=True)

                for epoch in tqdm(range(args.epochs)):
                    i_iter += 1
                    epoch_losses = []
                    for confounder, outcome, uid in train_loader_shuffled:
                        model.train()
                        # train IPW
                        optimizer.zero_grad()

                        dx, sex, age = confounder[0], confounder[1], confounder[2]
                        dx = torch.sum(dx, 1)
                        dx = torch.where(dx > 0, 1., 0.)
                        X = torch.cat((dx, sex.unsqueeze(1), age), 1)
                        Y = outcome[:, 0]
                        if args.cuda:
                            X = X.float().to('cuda')
                            Y = Y.long().to('cuda')

                        Y_logits = model(X)
                        # loss_ipw = F.binary_cross_entropy_with_logits(treatment_logits, treatment.float())
                        loss = F.cross_entropy(Y_logits, Y)
                        loss.backward()
                        optimizer.step()
                        epoch_losses.append(loss.item())

                    # just finish 1 epoch
                    # scheduler.step()
                    epoch_losses = np.mean(epoch_losses)

                    auc_val, loss_val, Y_val, Y_pred_val, uid_val, X_val = transfer_data(model,
                                                                                         val_loader,
                                                                                         cuda=args.cuda,
                                                                                         normalized=False)
                    results.append(
                        (i_iter, i, epoch, hyper_paras, epoch_losses, loss_val, auc_val))

                    print('HP-i:{}, epoch:{}, train-loss:{}, val-loss:{}, val-auc:{}'.format(
                        i, epoch, epoch_losses, loss_val, auc_val))

                    if auc_val > best_auc:
                        best_model = model
                        best_hyper_paras = hyper_paras
                        best_auc = auc_val
                        best_model_epoch = epoch
                        print('Save Best PSModel at Hyper-iter[{}/{}]'.format(i, len(hyper_paras_list)),
                              'Epoch: ', epoch, 'val-auc:', best_auc)
                        print(hyper_paras_names)
                        print(hyper_paras)
                        save_model(model, args.save_model_filename, model_params=model_params)

            col_name = ['i', 'ipara', 'epoch', 'paras', "train_epoch_losses", "loss_val", "auc_val"]
            results = pd.DataFrame(results, columns=col_name)

            print('Model selection finished! Save Global Best PSModel at Hyper-iter [{}/{}], Epoch: {}'.format(
                i, len(hyper_paras_list), best_model_epoch), 'val-auc:', best_auc)
            print(hyper_paras_names)
            print(best_hyper_paras)
            results.to_csv(args.save_model_filename + '_ALL-model-select.csv')

            # evaluation on test
            best_model = load_model(mlp.MLP, args.save_model_filename)
            best_model.to(args.device)
            test_auc, test_loss, test_y, test_y_pre, test_uid, test_x = transfer_data(best_model, test_loader,
                                                                                       cuda=args.cuda,
                                                                                       normalized=False)
            print('test_loss:', test_loss, 'test_auc: ', test_auc)
            print('...Original data results: ')
            print('......Results at specificity 0.9:')
            result_1 = ml.MLModels._performance_at_specificity_or_threshold(test_y_pre, test_y, specificity=0.9)
            print('......Results at specificity 0.95:')
            result_2 = ml.MLModels._performance_at_specificity_or_threshold(test_y_pre, test_y, specificity=0.95)

            df1 = pd.DataFrame([result_1 + (best_hyper_paras,), result_2 + (best_hyper_paras,)],
                               columns=["AUC", "threshold", "Specificity", "Sensitivity/recall", "PPV/precision",
                                        "n_negative", "n_positive", "precision_recall_fscore_support",
                                        'best_hyper_paras'],
                               index=['r_9', 'r_95'])
            df1.to_csv('output/test_results_{}r{}.csv'.format(args.run_model, args.random_seed))
            # df2 = pd.DataFrame({'aux_x': aux_x.mean(axis=0), 'train_x': train_x.mean(axis=0),
            #                     'train_x_pos': train_x[train_y == 1].mean(axis=0)},
            #                    index=feature_name).reset_index()
            # df2.to_csv('output/data_train_vs_aux_{}r{}.csv'.format(args.run_model, args.random_seed))

            threshold = result_2[1]
            test_y_pre = (test_y_pre > threshold).astype(int)
            r = precision_recall_fscore_support(test_y, test_y_pre)
            print('precision_recall_fscore_support:\n', r)
            feat = [';'.join(feature_name[np.nonzero(test_x[i, :])[0]]) for i in range(len(test_x))]
            pd.DataFrame(
                {'test_uid': test_uid, 'test_y_pre_prob': test_y_pre, 'test_y_pre': test_y_pre, 'test_y': test_y,
                 'feat': feat}).to_csv('output/test_pre_details_{}r{}.csv'.format(args.run_model, args.random_seed))
            print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    if args.run_model in ['LR', 'XGBOOST', 'LIGHTGBM']:
        print("**************************************************")
        print("**************************************************")
        print(args.run_model, ' model learning:')
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

        ## SVD can improve performance, due to sparse matrix
        # svd = TruncatedSVD(n_components=36, random_state=args.random_seed)
        # svd.fit(train_x)
        # train_x = svd.transform(train_x)
        # val_x = svd.transform(val_x)
        # test_x = svd.transform(test_x)
        # aux_x = svd.transform(aux_x)

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
                'min_child_samples': [50, 100, 150, 200, 250],
                'random_state': [args.random_seed],
            }
        #

        model = ml.MLModels(args.run_model, paras_grid).fit(train_x, train_y, val_x, val_y)

        sample_weight = np.concatenate((np.ones_like(train_y), 1 * np.ones_like(aux_y)))
        model_aux = ml.MLModels(args.run_model, paras_grid).fit(
            np.concatenate((train_x, aux_x)),
            np.concatenate((train_y, aux_y)), val_x, val_y, sample_weight=sample_weight)

        print('...Original data results: ')
        print('......Results at specificity 0.9:')
        result_1 = model.performance_at_specificity_or_threshold(test_x, test_y, specificity=0.9)
        print('......Results at specificity 0.95:')
        result_2 = model.performance_at_specificity_or_threshold(test_x, test_y, specificity=0.95)
        print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

        print('...Using aux data results: ')
        print('......Results at specificity 0.9:')
        result_aux_1 = model_aux.performance_at_specificity_or_threshold(test_x, test_y, specificity=0.9)
        print('......Results at specificity 0.95:')
        result_aux_2 = model_aux.performance_at_specificity_or_threshold(test_x, test_y, specificity=0.95)

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
        feat = [';'.join(feature_name[np.nonzero(test_x[i, :])[0]]) for i in range(len(test_x))]
        pd.DataFrame({'test_uid': test_uid, 'test_y_pre_prob': test_y_pre_prob, 'test_y_pre': test_y_pre, 'test_y': test_y,
                      'feat': feat}).to_csv('output/test_pre_details_{}r{}.csv'.format(args.run_model, args.random_seed))

        ### debug: print features of one patient:
        with open(r'pickles/final_pats_1st_neg_dict.pkl', 'rb') as f:
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

        uid = '937504'
        arecord = datadict.get(uid)
        a_details = []
        for a in arecord:
            a_details.append(a[:4] + [x + '_' + icd_des.get(x, '') for x in a[4:]])
        for a in a_details:
            print(a)

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
