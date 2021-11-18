import sys

# for linux env.
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import svm, tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import itertools
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, roc_auc_score, \
    confusion_matrix
import time
from sklearn.metrics import log_loss
from tqdm import tqdm
import xgboost as xgb
import lightgbm as lgb


class MLModels:
    """
    Designed for binary classification, automatic searching hyper-parameter space
    """
    def __init__(self, learner, paras_grid=None):
        self.learner = learner
        assert self.learner in ('LR', 'XGBOOST', 'LIGHTGBM', 'SVM', 'KNN')

        if (paras_grid is None) or (not paras_grid) or (not isinstance(paras_grid, dict)):
            self.paras_grid = {}
        else:
            self.paras_grid = {k: v for k, v in paras_grid.items()}
            for k, v in self.paras_grid.items():
                if isinstance(v, str):
                    print(k, v, 'is a fixed parameter')
                    self.paras_grid[k] = [v, ]

        if self.paras_grid:
            paras_names, paras_v = zip(*self.paras_grid.items())
            paras_list = list(itertools.product(*paras_v))
            self.paras_names = paras_names
            self.paras_list = [{self.paras_names[i]: para[i] for i in range(len(para))} for para in paras_list]
        else:
            self.paras_names = []
            self.paras_list = [{}]

        self.best_hyper_paras = None
        self.best_model = None
        self.best_val = float('-inf')

        self.results = []

    def fit(self, X_train, Y_train, X_val, Y_val, verbose=1, sample_weight=None):
        start_time = time.time()
        if verbose:
            print('Model {} Searching Space N={}: '.format(self.learner, len(self.paras_list)), self.paras_grid)

        for para_d in tqdm(self.paras_list):
            if self.learner == 'LR':
                if para_d.get('penalty', '') == 'l1':
                    para_d['solver'] = 'liblinear'
                elif para_d.get('penalty', '') == 'l2':
                    para_d['solver'] = 'lbfgs'
                else:
                    para_d['solver'] = 'saga'
                model = LogisticRegression(**para_d).fit(X_train, Y_train, sample_weight=sample_weight)
            elif self.learner == 'XGBOOST':
                model = xgb.XGBClassifier(**para_d).fit(X_train, Y_train)
            elif self.learner == 'LIGHTGBM':
                model = lgb.LGBMClassifier(**para_d).fit(X_train, Y_train)
            elif self.learner == 'SVM':
                model = svm.SVC(**para_d).fit(X_train, Y_train)
            elif self.learner == 'KNN':
                model = KNeighborsClassifier(**para_d).fit(X_train, Y_train)
            else:
                raise ValueError

            T_val_predict = model.predict_proba(X_val)[:, 1]
            auc_val = roc_auc_score(Y_val, T_val_predict)
            T_train_predict = model.predict_proba(X_train)[:, 1]
            loss_train = log_loss(Y_train, T_train_predict)

            self.results.append((para_d, loss_train, auc_val))  # model,  not saving model for less disk

            if auc_val > self.best_val:  #
                self.best_model = model
                self.best_hyper_paras = para_d
                self.best_val = auc_val

        self.results = pd.DataFrame(self.results, columns=['paras', 'train_loss', 'validation_auc'])
        if verbose:
            self.report_stats()

        print('Fit Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
        return self

    def report_stats(self):
        print('Model {} Searching Space N={}: '.format(self.learner, len(self.paras_list)), self.paras_grid)
        print('Best model: ', self.best_model)
        print('Best configuration: ', self.best_hyper_paras)
        print('Best fit value ', self.best_val)
        pd.set_option('display.max_columns', None)
        describe = self.results.describe()
        print('AUC stats:\n', describe)
        return describe

    def predict_proba(self, X):
        pred = self.best_model.predict_proba(X)[:, 1]
        return pred

    def predict_loss(self, X, T):
        T_pre = self.predict_proba(X)
        return log_loss(T, T_pre)

    @staticmethod
    def _performance_at_specificity_or_threshold_old(y_pre_prob, y, specificity=0.95, threshold=None, tolerance=1e-5, maxiter=100, verbose=1):

        if threshold is None:
            threshold = 0.5
            left = 1e-6
            right = 1.
            i = 0
            while (left < right) and (i <= maxiter):
                i += 1
                y_pre = (y_pre_prob > threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(y, y_pre).ravel()
                spec = tn / (tn + fp)  # recall of the negative class
                if np.abs(spec - specificity) < tolerance:
                    break

                if spec > specificity:
                    right = threshold
                elif spec < specificity:
                    left = threshold

                threshold = (left + right) / 2.0

        auc = roc_auc_score(y, y_pre_prob)
        y_pre = (y_pre_prob > threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, y_pre).ravel()
        specificity = tn / (tn + fp)  # recall of the negative class
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)  # recall of the positive class, specificity
        r = precision_recall_fscore_support(y, y_pre)
        if verbose:
            print('auc:{:5f}\tthreshold:{:.5f}\tspecificity:{:5f}\tSensitivity/Recall:{:5f}\t'
                  'PPV/Precision:{:5f}\t#negative:{}\t#positive:{}'.format(
                auc, threshold, specificity, recall, precision, r[3][0], r[3][1]))
            # print('precision_recall_fscore_support:\n', r)

        return auc, threshold, specificity, recall, precision, r[3][0], r[3][1], r

    @staticmethod
    def _performance_at_specificity_or_threshold(y_pre_prob, y, specificity=0.95, threshold=None, tolerance=1e-6,
                                                 maxiter=100, verbose=1):

        sort_index = np.argsort(y_pre_prob)[::-1]
        y_pre_prob_sorted, y_sorted = y_pre_prob[sort_index], y[sort_index]
        mid = None
        results = []
        if threshold is None:  # then find mid
            left = 0
            right = len(y_pre_prob)-1
            i = 0
            while (left <= right) and (i <= maxiter):
                i += 1
                mid = (left + right) // 2
                threshold = y_pre_prob_sorted[mid]
                y_pre = np.ones_like(y)
                y_pre[mid:] = 0
                tn, fp, fn, tp = confusion_matrix(y_sorted, y_pre).ravel()
                spec = tn / (tn + fp)  # recall of the negative class
                if verbose > 1:
                    print(i, mid, threshold, spec)
                results.append([i, mid, threshold, spec])
                if np.abs(spec - specificity) < tolerance:
                    break
                if spec < specificity:
                    right = mid - 1
                elif spec > specificity:
                    left = mid+1
                else:
                    break

        if len(results) >= 2:
            adjusted_id = np.argmin([np.abs(results[-1][3] - specificity), np.abs(results[-2][3] - specificity)])
            mid = results[-1-adjusted_id][1]

        if mid is not None:
            y_pre = np.ones_like(y)
            y_pre[mid:] = 0
            y_truth = y_sorted
        else:
            y_pre = (y_pre_prob > threshold).astype(int)
            y_truth = y

        auc = roc_auc_score(y, y_pre_prob)
        tn, fp, fn, tp = confusion_matrix(y_truth, y_pre).ravel()
        spec = tn / (tn + fp)  # recall of the negative class
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)  # recall of the positive class, specificity
        r = precision_recall_fscore_support(y_truth, y_pre)

        if verbose:
            print('auc:{:5f}\tthreshold:{:.5f} ({}th in {})\tSpecificity:{:5f}\tSensitivity/Recall:{:5f}\t'
                  'PPV/Precision:{:5f}\t#negative:{}\t#positive:{}'.format(
                auc, threshold, mid, len(y_pre_prob), spec, recall, precision, r[3][0], r[3][1]))
            # print('precision_recall_fscore_support:\n', r)

        return auc, threshold, spec, recall, precision, r[3][0], r[3][1], r

    def performance_at_specificity_or_threshold(self, x, y, specificity=0.95, threshold=None,
                                                tolerance=1e-6, maxiter=100, verbose=1):
        y_pre_prob = self.predict_proba(x)

        return self._performance_at_specificity_or_threshold(y_pre_prob, y,
                                                             specificity=specificity,
                                                             threshold=threshold,
                                                             tolerance=tolerance,
                                                             maxiter=maxiter,
                                                             verbose=verbose)

