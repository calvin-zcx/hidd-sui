"""
_*_ coding: utf-8 _*_
@ Author: Yu Hou
@ File: dataset_lstm.py
@ Time: 5/31/22 3:07 PM
"""

import numpy as np
import torch.utils.data
from Vocab import *
from tqdm import tqdm
import copy
import pandas as pd


class LSTM_Dataset(torch.utils.data.Dataset):
    def __init__(self, patient_list, diag_code_threshold=None, diag_code_topk=None, diag_name=None, diag_code_vocab=None,
                 diag_visit_max_length_quntile=0.99):
        self.patient_list = patient_list
        self.diagnoses_visits = []
        self.sexes = []
        self.ages = []
        self.outcome = []
        self.uid = []

        print('Build Dataset:......')
        print('diag_code_threshold:',diag_code_threshold,
              'diag_code_topk:', diag_code_topk,
              'diag_code_vocab:', diag_code_vocab)

        print('Extract data features from compressed list')
        for uid, patient_confounder, patient_outcome in tqdm(self.patient_list):
            self.outcome.append(patient_outcome)
            diag_visit, age, sex, _n_sequence, _time_diff_1andlast = patient_confounder
            self.diagnoses_visits.append(diag_visit)
            self.sexes.append(sex)
            self.ages.append(age)
            self.uid.append(uid)

        print('Build diag code vocabulary')
        if diag_code_vocab is None:
            self.diag_code_vocab = CodeVocab(diag_code_threshold, diag_code_topk, diag_name)
            self.diag_code_vocab.add_patients_visits(self.diagnoses_visits)
        else:
            self.diag_code_vocab = diag_code_vocab

        print('Created Diagnoses Vocab: %s' % self.diag_code_vocab)
        self.visit_seq_len = pd.DataFrame([len(patient_visit) for patient_visit in self.diagnoses_visits])
        print('self.visit_seq_len:', self.visit_seq_len.describe())
        print('Using diag_visit_max_length_quntile:', diag_visit_max_length_quntile)
        assert 0 < diag_visit_max_length_quntile <= 1
        self.diag_visit_max_length = int(np.quantile(self.visit_seq_len, diag_visit_max_length_quntile))  # max(self.visit_seq_len)
        self.diag_vocab_length = len(self.diag_code_vocab)
        print('Diagnoses Visit Max Length: %d' % self.diag_visit_max_length)

        print('Encoding age')
        self.ages = self._process_ages()

        print('Encoding outcome')
        # newly added 2021-10-25
        self.outcome = self._process_outcome()

        # feature name
        diag_col_name = self.diag_code_vocab.feature_name()
        self.col_name = (diag_col_name, ['sex'], ['age10-14', 'age15-19', 'age20-24'])
        self.FEATURE_NAME = np.asarray(sum(self.col_name, []))

        self.DIM_OF_CONFOUNDERS = len(self.FEATURE_NAME)
        print('DIM_OF_CONFOUNDERS: ', self.DIM_OF_CONFOUNDERS)
        print('LSTM_Dataset initialization done!')

    def _process_visits(self, visits, max_len_visit, vocab):
        res = np.zeros((max_len_visit, len(vocab)))
        if len(visits) > max_len_visit:
            delta = len(visits) - max_len_visit
            combined_left = list(set(sum(visits[0:(delta+1)], [])))
            visits = [combined_left, ] + visits[delta+1:]
        for i, visit in enumerate(visits):
            res[i] = self._process_code(vocab, visit)
        # col_name = [vocab.id2name.get(x, '') for x in range(len(vocab))]
        return res  # , col_name

    def _process_code(self, vocab, codes):
        multi_hot = np.zeros((len(vocab, )), dtype='float')
        for code_ in codes:
            code = code_.split('_')[0]
            if code in vocab.code2id:
                multi_hot[vocab.code2id[code]] = 1
        return multi_hot

    def _process_ages(self):
        ages = np.zeros((len(self.ages), 3))
        n_out = 0
        for i, x in enumerate(self.ages):
            # if 10 <= x <= 14:
            if x <= 14: # change 2022-08-01
                ages[i, 0] = 1
            elif 15 <= x <= 19:
                ages[i, 1] = 1
            elif 20 <= x <= 24:
                ages[i, 2] = 1
            else:  # >= 24
                # print(i, 'row, wrong age within [10, 24]: ', x)
                ages[i, 2] = 1
                n_out += 1
                # raise ValueError
        print('wrong age within [10, 24]: {}/{}'.format(n_out, len(self.ages)))
        return ages

    def _process_outcome(self):
        # outcome:
        # 0: two or more events, w/o suicide, negative
        # 1: two or more events, last is suicide, positive
        # 2: only 1 visit w/o suicide
        # 3:  first visit with suicide, only use first visit, may change pre_feat.py later
        outcome = np.zeros((len(self.outcome), 5))
        for i, o in enumerate(self.outcome):
            x, t2e = o
            if x == 0:
                outcome[i, 0] = 1
            elif x == 1:
                outcome[i, 1] = 1
            elif x == 2:
                outcome[i, 2] = 1
            elif x == 3:
                outcome[i, 3] = 1
            else:
                print(i, 'row, wrong outcome (should in 0,1,-1,2): ', x)
                raise ValueError

            outcome[i, 4] = t2e

        return outcome

    def __getitem__(self, index):
        # return labels
        outcome = self.outcome[index]
        Y = copy.deepcopy(outcome)
        Y[0] = Y[0] + Y[2]
        labels = Y[:2].argmax()
        diag = self.diagnoses_visits[index]

        if labels == 1:
            # For positive, leave out last suicide visit
            diag = diag[:-1]

        diag = self._process_visits(diag, self.diag_visit_max_length, self.diag_code_vocab)  # T_dx * D_dx

        sex = self.sexes[index]
        age = self.ages[index]

        confounder = (diag, sex, age)

        uid = self.uid[index]

        return confounder, labels, outcome, uid

    def __len__(self):
        return len(self.diagnoses_visits)