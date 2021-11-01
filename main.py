import sys
# for linux env.
import torch
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

print = functools.partial(print, flush=True)
from dataset import *
import random
import itertools
from model import ml, mlp, autoencoder
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, roc_auc_score, \
    confusion_matrix
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import torch.utils.data as data_utils
from sklearn.cluster import KMeans


def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    # Input
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument('--run_model', choices=['LSTM', 'LR', 'MLP', 'XGBOOST',
                                                'LIGHTGBM', "PRETRAIN", "MMLP"], default='MMLP')
    # Deep PSModels
    parser.add_argument('--batch_size', type=int, default=256)  # 768)  # 64)
    parser.add_argument('--learning_rate', type=float, default=1e-3)  # 0.001
    parser.add_argument('--weight_decay', type=float, default=1e-6)  # )0001)
    parser.add_argument('--epochs', type=int, default=35)  #40)  # 30
    # MLP
    parser.add_argument('--hidden_size', type=str, default='', help=', delimited integers')
    # Output
    parser.add_argument('--output_dir', type=str, default='output/')
    parser.add_argument('--pretrain_model_filename', type=str, default='output/S0_PRETRAIN')
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


if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    print('args: ', args)
    print('random_seed: ', args.random_seed)
    print('args.device: ', args.device)

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
    x, y, uid_list, y_more = my_dataset.flatten_to_tensor()
    my_dataset_aux = Dataset(data_1st_sui, diag_name=dx_name, diag_code_vocab=my_dataset.diag_code_vocab)
    x_aux, y_aux, uid_list_aux, y_more_aux = my_dataset_aux.flatten_to_tensor()

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
    dataset_size_aux = len(my_dataset_aux)
    indices = list(range(dataset_size))
    indices_aux = list(range(dataset_size, dataset_size + dataset_size_aux))

    train_index = int(np.floor(train_ratio * dataset_size))
    val_index = int(np.floor(val_ratio * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices, test_indices = indices[:train_index], \
                                               indices[train_index:train_index + val_index], \
                                               indices[train_index + val_index:]
    if args.run_model in ['PRETRAIN']:
        # xx = np.concatenate((x[:5000,:], x_aux[:200,:]))
        # yy = np.concatenate((y[:5000,:], y_aux[:200,:]))
        xx = np.concatenate((
            y_more[y[:, 1]==1, :], # suicide visit of positive
            x_aux  # 1st suicide visit
        ))
        yy = np.concatenate((y[y[:, 1]==1,:], y_aux))

        # feature_name_xx = feature_name[np.where(xx.sum(0) > 0)]
        # xx = xx[:, np.where(xx.sum(0)>0)[0]]
        tsne_plot(xx, yy, perplexity=50, dump=True, fname='tsne-raw-positiveSui-vs-1stSui')
        # kmeans = KMeans(n_clusters=3, random_state=args.random_seed).fit(xx)
        # kmeans.labels_
        # df = pd.DataFrame(xx, columns=feature_name_xx)
        # df['KM-label'] = kmeans.labels_
        # df.to_excel('positivelast_1stsui_cluster.xlsx')
        # df_center = pd.DataFrame(
        #     {'0': xx[np.where(kmeans.labels_==0)].mean(0),
        #      '1': xx[np.where(kmeans.labels_ == 1)].mean(0),
        #      '2': xx[np.where(kmeans.labels_ == 2)].mean(0)},
        #     index=feature_name_xx)
        # df_center.to_excel('positivelast_1stsui_clusterCenter.xlsx')

        # build AE training data, only include type 1, 2, 4. Not using 3 positive
        # index for 1,2,3 training data + 4 whole
        train_indices_AE = train_indices + indices_aux
        # index for 1,2,3 training data + 4 whole  -  2 training indices
        # train_indices_AE = list(set(train_indices_AE) - set(np.where((y[train_indices, 1]==1))[0]))
        # train_indices_AE = list(set(train_indices_AE) - set(np.where((y[train_indices, 2]==1))[0]))
        # train_indices_AE = train_indices_aux

        x_AE = np.concatenate((x, x_aux))
        y_AE = np.concatenate((y, y_aux))
        train_AE = data_utils.TensorDataset(torch.FloatTensor(x_AE), torch.LongTensor(y_AE[:,:4])) # not using Y last t2e dim
        train_sampler_AE = SubsetRandomSampler(train_indices_AE)
        train_loader = data_utils.DataLoader(train_AE, batch_size=args.batch_size, sampler=train_sampler_AE)

        model_params = dict(input_size=n_feature)
        print('Model: Autoencoder')
        print(model_params)
        premodel = autoencoder.Autoencoder(**model_params)
        if args.cuda:
            premodel = premodel.to('cuda')
        print(premodel)
        # criterion = F.binary_cross_entropy_with_logits(wx_pos, y) #nn.MSELoss()

        optimizer = torch.optim.Adam(premodel.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        i_iter = 0
        epoch_losses = []
        for epoch in tqdm(range(args.epochs)):
            i_iter += 1
            epoch_losses = []
            for X, Y in train_loader:
                if i_iter == 1:
                    print(X.shape, Y.shape)
                    print(X.dtype, Y.dtype)
                premodel.train()
                optimizer.zero_grad()
                if args.cuda:
                    X = X.to('cuda')  # X.float().to('cuda')
                    # Y = Y.long().to('cuda')

                Y_logits = premodel(X)
                loss = F.binary_cross_entropy_with_logits(Y_logits, X)
                # loss = F.cross_entropy(Y_logits, Y)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            # just finish 1 epoch
            # scheduler.step()
            epoch_losses = np.mean(epoch_losses)
            print('HP-i:{}, epoch:{}, train-loss:{}'.format(
                -1, epoch, epoch_losses))

        save_model(premodel, args.save_model_filename, model_params=model_params)

        xx = torch.FloatTensor(xx)
        model = premodel.to('cpu')
        with torch.no_grad():
            premodel.eval()
            embedding = premodel.encoder(xx)
            embedding = embedding.detach().data.numpy()
        tsne_plot(embedding, yy, perplexity=50, dump=True, fname='tsne-AE-positiveSui-vs-1stSui')

    if args.run_model in ['MLP', 'AE', 'MMLP']:
        # Try to use top y_more codes for multi-task/or sequence-2-sequence
        print('y_more, non-zero: ', (y_more.sum(0) > 0).sum(), ' ratio:', (y_more.sum(0) > 0).mean())
        multi_feature_number_threshold = 20
        multi_dim = np.where(y_more.sum(0) > multi_feature_number_threshold)[0]  # at least 20 patients for one feature dim
        multi_dim_name = my_dataset.FEATURE_NAME[multi_dim]
        n_multi_dim = len(multi_dim)
        print('Use #features >={} for multi-task learning, {} features are: '.format(
            multi_feature_number_threshold, len(multi_dim)), multi_dim_name)
        my_dataset_flat = data_utils.TensorDataset(
            torch.FloatTensor(x),  # feature
            torch.LongTensor(y[:, :4]),  # Y, data type
            torch.FloatTensor(y[:, 4]),  # Y, time 2 event
            torch.FloatTensor(y_more[:, multi_dim]),  # Y, selected y for multi-task
            torch.LongTensor(range(dataset_size)),  # index of data, to get uid_list[idx]
        )
        exclude_1_visit = False
        if exclude_1_visit:
            train_indices = list(set(train_indices) - set(np.asarray(indices)[np.where((y[indices, 2] == 1))[0]]))
            val_indices = list(set(val_indices) - set(np.asarray(indices)[np.where((y[indices, 2] == 1))[0]]))
            test_indices = list(set(test_indices) - set(np.asarray(indices)[np.where((y[indices, 2] == 1))[0]]))

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_loader = torch.utils.data.DataLoader(my_dataset_flat, batch_size=args.batch_size,
                                                   sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(my_dataset_flat, batch_size=args.batch_size,
                                                 sampler=val_sampler)
        test_loader = torch.utils.data.DataLoader(my_dataset_flat, batch_size=args.batch_size,
                                                  sampler=test_sampler)
        if args.run_model == 'MMLP':
            print("**************************************************")
            print("**************************************************")
            print(args.run_model, ' model learning:')
            # PSModels configuration & training
            paras_grid = {
                'hidden_size': [[32,32], [64,32], [128, 64], [128, 32]],  #, 32,  64, 128], [32,32], [64,32],[128, 64], [256,128], [128, 64, 32]
                'lr': [1e-3, 1e-4],
                'weight_decay': [1e-4, 1e-5, 1e-6],
                'batch_size': [1024],
                'dropout': [0.5],
                'multi_task_classes': [n_multi_dim]
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
                hidden_size, lr, weight_decay, batch_size, dropout, multi_task_classes = hyper_paras
                print('In hyper-paras space [{}/{}]...'.format(i, len(hyper_paras_list)))
                print(hyper_paras_names)
                print(hyper_paras)

                train_loader_shuffled = torch.utils.data.DataLoader(my_dataset_flat, batch_size=batch_size,
                                                                    sampler=train_sampler)
                print('len(train_loader_shuffled): ', len(train_loader_shuffled),
                      'train_loader_shuffled.batch_size: ', train_loader_shuffled.batch_size)

                model_params = dict(input_size=n_feature, num_classes=2, hidden_size=hidden_size,
                                    dropout=dropout, multi_task_classes=multi_task_classes)
                print('Model: ')
                print(model_params)
                model = mlp.MultiMLP(**model_params)
                if args.cuda:
                    model = model.to('cuda')
                print(model)

                optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
                # optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

                for epoch in tqdm(range(args.epochs)):
                    i_iter += 1
                    epoch_losses = []
                    for X, Y, Y_t2e, Y_more, idx in train_loader_shuffled:
                        model.train()
                        # train IPW
                        optimizer.zero_grad()

                        if args.cuda:
                            X = X.float().to('cuda')
                            Y = Y.long().to('cuda')
                            Y_more = Y_more.float().to('cuda')
                        # How to use Y?
                        # a. Ignore Y[:, 2]. If not using 1 visit, Y[:, :2].
                        # b. Treat  Y[:, 2] as negative. If using 1 visit, Y[:, 0] = Y[:, 0] + Y[:, 2]
                        # c. Treat  Y[:, 2] as censoring. Or to predict Y[:, 2] censoring?
                        if not exclude_1_visit:
                            Y[:, 0] = Y[:, 0] + Y[:, 2]
                        _, labels = Y[:, :2].max(dim=1)

                        Y_logits, Y2_logits, Y3_logits = model(X)
                        # loss_ipw = F.binary_cross_entropy_with_logits(treatment_logits, treatment.float())
                        # loss1 = F.cross_entropy(Y_logits, Y)
                        loss_sup = nn.CrossEntropyLoss()(Y_logits, labels)
                        loss_multitask = F.binary_cross_entropy_with_logits(Y2_logits, Y_more)
                        loss_AE = F.binary_cross_entropy_with_logits(Y3_logits, X)

                        # loss = loss1 + loss2
                        loss = loss_sup + 0.*loss_multitask + 0.*loss_AE  # + loss_multitask  #  # + 0.5*loss_multitask  # + 0.5*loss_AE
                        loss.backward()
                        optimizer.step()
                        epoch_losses.append(loss.item())

                    # just finish 1 epoch
                    # scheduler.step()
                    epoch_losses = np.mean(epoch_losses)

                    auc_val, loss_val, Y_val, Y_pred_val, uid_val, X_val, X_val_embed = transfer_data(model,
                                                                                         val_loader,
                                                                                         cuda=args.cuda,
                                                                                         normalized=False,
                                                                                         pretrain_model=None,
                                                                                         exclude_1_visit = exclude_1_visit)
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
            best_model = load_model(mlp.MultiMLP, args.save_model_filename)
            best_model.to(args.device)
            test_auc, test_loss, test_y, test_y_pre, test_uid, test_x, test_x_embed = transfer_data(best_model, test_loader,
                                                                                       cuda=args.cuda,
                                                                                       normalized=False,
                                                                                       pretrain_model=None,
                                                                                       exclude_1_visit =exclude_1_visit)
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
                {'test_uid': uid_list[test_uid], 'test_y_pre_prob': test_y_pre, 'test_y_pre': test_y_pre, 'test_y': test_y,
                 'feat': feat}).to_csv('output/test_pre_details_{}r{}.csv'.format(args.run_model, args.random_seed))
            print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

        if args.run_model == 'MLP':
            print("**************************************************")
            print("**************************************************")
            print(args.run_model, ' model learning:')
            use_pretrain = False
            if use_pretrain:
                pretrain_model = load_model(autoencoder.Autoencoder, args.pretrain_model_filename)
                pretrain_model.to(args.device)
            else:
                pretrain_model = None
            # PSModels configuration & training
            paras_grid = {
                'hidden_size': [[32,32], [128, 64]],  #[128, 64], 32,  64, 128], [32,32], [64,32],[128, 64], [256,128], [128, 64, 32][128, 32]
                'lr': [1e-3, 1e-4],
                'weight_decay': [1e-4, 1e-5, 1e-6],
                'batch_size': [1024],
                'dropout': [0.5],
            }
            # paras_grid = {
            #     'hidden_size': [0, 8, 16, 32],  # , 64, 128],
            #     'lr': [1e-3],
            #     'weight_decay': [1e-6],
            #     'batch_size': [512],
            #     'dropout': [0.5],
            # }
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

                train_loader_shuffled = torch.utils.data.DataLoader(my_dataset_flat, batch_size=batch_size,
                                                                    sampler=train_sampler)
                print('len(train_loader_shuffled): ', len(train_loader_shuffled),
                      'train_loader_shuffled.batch_size: ', train_loader_shuffled.batch_size)
                if use_pretrain:
                    model_params = dict(input_size=pretrain_model.embedding_dim, num_classes=2, hidden_size=hidden_size, dropout=dropout, )
                else:
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
                    for X, Y, Y_t2e, Y_more, idx in train_loader_shuffled:
                        model.train()
                        # train IPW
                        optimizer.zero_grad()

                        if args.cuda:
                            X = X.float().to('cuda')
                            Y = Y.long().to('cuda')
                            Y_more = Y_more.long().to('cuda')
                        # How to use Y?
                        # a. Ignore Y[:, 2]. If not using 1 visit, Y[:, :2].
                        # b. Treat  Y[:, 2] as negative. If using 1 visit, Y[:, 0] = Y[:, 0] + Y[:, 2]
                        # c. Treat  Y[:, 2] as censoring. Or to predict Y[:, 2] censoring?
                        if not exclude_1_visit:
                            Y[:, 0] = Y[:, 0] + Y[:, 2]
                        _, labels = Y[:, :2].max(dim=1)
                        if use_pretrain:
                            with torch.no_grad():
                                pretrain_model.eval()
                                X_embed = pretrain_model.encoder(X)
                        else:
                            X_embed = X
                        Y_logits = model(X_embed)
                        # loss_ipw = F.binary_cross_entropy_with_logits(treatment_logits, treatment.float())
                        # loss1 = F.cross_entropy(Y_logits, Y)
                        loss1 = nn.CrossEntropyLoss()(Y_logits, labels)
                        # loss2 = F.binary_cross_entropy_with_logits(Y_logits, Y_more)
                        # loss = loss1 + loss2
                        loss = loss1
                        loss.backward()
                        optimizer.step()
                        epoch_losses.append(loss.item())

                    # just finish 1 epoch
                    # scheduler.step()
                    epoch_losses = np.mean(epoch_losses)

                    auc_val, loss_val, Y_val, Y_pred_val, uid_val, X_val, X_val_embed = transfer_data(model,
                                                                                         val_loader,
                                                                                         cuda=args.cuda,
                                                                                         normalized=False,
                                                                                         pretrain_model=pretrain_model,
                                                                                         exclude_1_visit = exclude_1_visit)
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
            test_auc, test_loss, test_y, test_y_pre, test_uid, test_x, test_x_embed = transfer_data(best_model, test_loader,
                                                                                       cuda=args.cuda,
                                                                                       normalized=False,
                                                                                       pretrain_model=pretrain_model,
                                                                                       exclude_1_visit=exclude_1_visit)
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
                {'test_uid': uid_list[test_uid], 'test_y_pre_prob': test_y_pre, 'test_y_pre': test_y_pre, 'test_y': test_y,
                 'feat': feat}).to_csv('output/test_pre_details_{}r{}.csv'.format(args.run_model, args.random_seed))
            print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    if args.run_model in ['LR', 'XGBOOST', 'LIGHTGBM']:
        print("**************************************************")
        print("**************************************************")
        print(args.run_model, ' model learning:')

        def return_label(Y):
            # 1-visit-w/o-sui patient as negative
            Y[:, 0] = Y[:, 0] + Y[:, 2]
            return np.argmax(Y[:, :2], axis=1)
        print('Train data:')
        train_x, train_y, train_uid, train_y_more = x[train_indices], y[train_indices], uid_list[train_indices], y_more[train_indices]
        train_y = return_label(train_y)
        print('... pos ratio:', train_y.mean())
        print('Validation data:')
        val_x, val_y, val_uid, val_y_more = x[val_indices], y[val_indices], uid_list[val_indices], y_more[val_indices]
        val_y = return_label(val_y)
        print('... pos ratio:', val_y.mean())
        print('Test data:')
        test_x, test_y, test_uid, test_y_more = x[test_indices], y[test_indices], uid_list[test_indices], y_more[test_indices]
        test_y = return_label(test_y)
        print('... pos ratio:', test_y.mean())

        # If ? SVD can improve performance due to sparse matrix?
        # When using behaviour features, seems NO.
        # svd = TruncatedSVD(n_components=64, random_state=args.random_seed)
        # svd.fit(train_x)
        # train_x = svd.transform(train_x)
        # val_x = svd.transform(val_x)
        # test_x = svd.transform(test_x)

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
        else:
            raise ValueError
        #

        model = ml.MLModels(args.run_model, paras_grid).fit(train_x, train_y, val_x, val_y)

        print('...Original data results: ')
        print('......Results at specificity 0.9:')
        result_1 = model.performance_at_specificity_or_threshold(test_x, test_y, specificity=0.9)
        print('......Results at specificity 0.95:')
        result_2 = model.performance_at_specificity_or_threshold(test_x, test_y, specificity=0.95)

        df1 = pd.DataFrame([result_1 + (model.best_hyper_paras,), result_2 + (model.best_hyper_paras,),
                            ],
                           columns=["AUC", "threshold", "Specificity", "Sensitivity/recall", "PPV/precision",
                                    "n_negative", "n_positive", "precision_recall_fscore_support", 'best_hyper_paras'],
                           index=['r_9', 'r_95'])
        df1.to_csv('output/test_results_{}r{}.csv'.format(args.run_model, args.random_seed))

        if args.run_model == 'LR':
            df3 = pd.DataFrame({'train_x_model': model.best_model.coef_[0]},
                               index=feature_name).reset_index()
            df3.to_csv('output/model_coef_train_LRr{}.csv'.format(args.random_seed))

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

        # pickle.dump(model, open('pickles/models.pkl', 'wb'))


