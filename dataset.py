# import logging
import numpy as np
import torch.utils.data
from vocab import *
from tqdm import tqdm
from collections import Counter

# logger = logging.getLogger()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, patient_list, diag_code_threshold=None, diag_code_topk=None, diag_name=None, diag_code_vocab=None):
        # diag_code_vocab=None, med_code_vocab=None,
        self.patient_list = patient_list
        self.diagnoses_visits = []
        self.sexes = []
        self.ages = []
        self.outcome = []
        self.uid = []
        self.n_sequence = []
        self.time_diff_1andlast = []

        print('Build Dataset:......')
        print('diag_code_threshold:',diag_code_threshold,
              'diag_code_topk:', diag_code_topk,
              'diag_code_vocab:', diag_code_vocab)

        for uid, patient_confounder, patient_outcome in tqdm(self.patient_list):
            self.outcome.append(patient_outcome)
            diag_visit, age, sex, _n_sequence, _time_diff_1andlast = patient_confounder
            self.diagnoses_visits.append(diag_visit)
            self.sexes.append(sex)
            self.ages.append(age)
            self.uid.append(uid)
            self.n_sequence.append(_n_sequence)
            self.time_diff_1andlast.append(_time_diff_1andlast)

        if diag_code_vocab is None:
            self.diag_code_vocab = CodeVocab(diag_code_threshold, diag_code_topk, diag_name)
            self.diag_code_vocab.add_patients_visits(self.diagnoses_visits)
        else:
            self.diag_code_vocab = diag_code_vocab

        print('Created Diagnoses Vocab: %s' % self.diag_code_vocab)
        self.diag_visit_max_length = max([len(patient_visit) for patient_visit in self.diagnoses_visits])
        self.diag_vocab_length = len(self.diag_code_vocab)
        print('Diagnoses Visit Max Length: %d' % self.diag_visit_max_length)

        self.ages = self._process_ages()
        # newly added 2021-10-25
        self.outcome = self._process_outcome()
        print('self.n_sequence distribution:\n', Counter(self.n_sequence).most_common())
        self.n_sequence = self._process_n_sequence()
        print('np.ptp(self.time_diff_1andlast):', np.ptp(self.time_diff_1andlast),
              'year:', np.ptp(self.time_diff_1andlast) / 365)
        self.time_diff_1andlast = (self.time_diff_1andlast - np.min(self.time_diff_1andlast)) / np.ptp(
            self.time_diff_1andlast)
        self.time_diff_1andlast[np.isnan(self.time_diff_1andlast)] = 0

        # feature name
        diag_col_name = self.diag_code_vocab.feature_name()
        self.col_name = (diag_col_name,
                    ['sex'],
                    ['age10-14', 'age15-19', 'age20-24'],
                    ['n_sequence1', 'n_sequence2', 'n_sequence3', 'n_sequence4', 'n_sequence5orMore', 'time_diff_1andlast'])
        self.FEATURE_NAME = np.asarray(sum(self.col_name, []))

        # feature dim: diag_visit, 3age, 1sex, 5_n_sequence, 1_time_diff_1andlast
        # self.DIM_OF_CONFOUNDERS = len(self.diag_code_vocab) + 10
        self.DIM_OF_CONFOUNDERS = len(self.FEATURE_NAME)
        print('DIM_OF_CONFOUNDERS: ', self.DIM_OF_CONFOUNDERS)

    def _process_visits(self, visits, max_len_visit, vocab, vary_length=False):
        if vary_length:
            res = np.zeros((len(visits), len(vocab)))
        else:
            res = np.zeros((max_len_visit, len(vocab)))
        for i, visit in enumerate(visits):
            res[i] = self._process_code(vocab, visit)
        # col_name = [vocab.id2name.get(x, '') for x in range(len(vocab))]
        return res  # , col_name

    def _process_code(self, vocab, codes):
        multi_hot = np.zeros((len(vocab, )), dtype='float')
        for code in codes:
            code_ = code.split('_')[0]
            if code_ in vocab.code2id:
                multi_hot[vocab.code2id[code_]] = 1
        return multi_hot

    def _process_ages(self):
        ages = np.zeros((len(self.ages), 3))
        n_out = 0
        for i, x in enumerate(self.ages):
            if 10 <= x <= 14:
                ages[i, 0] = 1
            elif 15 <= x <= 19:
                ages[i, 1] = 1
            elif 20 <= x <= 24:
                ages[i, 2] = 1
            else:
                # print(i, 'row, wrong age within [10, 24]: ', x)
                ages[i, 2] = 1
                n_out += 1
                # raise ValueError
        print('wrong age within [10, 24]: {}/{}'.format(n_out, len(self.ages)))
        return ages

    def _process_n_sequence(self):
        n_sequence = np.zeros((len(self.n_sequence), 5))
        for i, x in enumerate(self.n_sequence):
            if x == 1:
                n_sequence[i, 0] = 1
            elif x == 2:
                n_sequence[i, 1] = 1
            elif x == 3:
                n_sequence[i, 2] = 1
            elif x == 4:
                n_sequence[i, 3] = 1
            elif x >= 5:
                n_sequence[i, 4] = 1
            else:
                print(i, 'row, wrong n_sequence >=1: ', x)
                raise ValueError
        return n_sequence

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
        print(
            outcome[:, 0].sum(),  ':two or more events, w/o suicide, negative\n',
            outcome[:, 1].sum(),  ':two or more events, last is suicide, positive\n',
            outcome[:, 2].sum(),  ':only 1 visit w/o suicide\n',
            outcome[:, 3].sum(),  ':first visit with suicide, only use first visit, may change pre_feat.py later\n',
            outcome[:, 4].mean(), ':mean time (days) to event/censoring'
        )

        return outcome

    def __getitem__(self, index):
        # Problem: very sparse due to 1. padding a lots of 0, 2. original signals in high-dim.
        # paddedsequence for 1 and graph for 2?
        # should give new self._process_visits and self._process_visits
        # also add more demographics for confounder
        diag = self.diagnoses_visits[index]
        diag = self._process_visits(diag, self.diag_visit_max_length, self.diag_code_vocab, vary_length=True)  # T_dx * D_dx

        sex = self.sexes[index]
        age = self.ages[index]
        # outcome = self.outcome[index][self.outcome_type]  # no time2event using in the matter rising
        outcome = self.outcome[index]

        n_sequence = self.n_sequence[index]
        time_diff_1andlast = self.time_diff_1andlast[index]
        confounder = (diag, sex, age, n_sequence, time_diff_1andlast)

        uid = self.uid[index]

        return confounder, outcome, uid

    def __len__(self):
        return len(self.diagnoses_visits)

    def flatten_to_tensor(self, use_behavior=False, normalized_count=True):
        print('Dataset flatten_to_tensor:......')
        print('use_behavior:', use_behavior, 'normalized_count: ', normalized_count)
        # 2021/10/25
        # for static, pandas dataframe-like learning
        # refer to flatten_data function
        # each individual's outcome[i]:
        #  0: two or more events, w/o suicide, negative
        #  1: two or more events, last is suicide, positive
        #  2: only 1 visit w/o suicide
        #  3: first visit with suicide, only use first visit, may change pre_feat.py later
        #  4: time to event/censoring
        x, y = [], []
        uid_list = []
        y_more = []
        for index in range(self.__len__()):
            confounder, outcome, uid = self.__getitem__(index)
            diag, sex, age, n_sequence, time_diff_1andlast = confounder

            if outcome[1]:
                dx = np.sum(diag[:-1], axis=0)
                # # issue: age is not suicide age, n)sequence is problem --> should only predict dx
                # x_at_sui.append(np.concatenate((diag[-1], [sex], age, n_sequence, [time_diff_1andlast])))
                y_more.append(diag[-1])
                # y_more.append(np.concatenate((diag[-1], [sex], age)))
            else:
                dx = np.sum(diag, axis=0)
                y_more.append(np.zeros_like(dx))
                # y_more.append(np.zeros_like(np.concatenate((diag[-1], [sex], age))))
            # encode as boolean value
            if not normalized_count:
                dx = np.where(dx > 0, 1, 0)
            if use_behavior:
                x.append(np.concatenate((dx, [sex], age, n_sequence, [time_diff_1andlast])))  #
            else:
                x.append(np.concatenate((dx, [sex], age)))  #
            y.append(outcome)
            uid_list.append(uid)

        x, y = np.asarray(x), np.asarray(y)
        if normalized_count:
            ndiag = diag.shape[1]
            x[:, :ndiag] = np.log(x[:, :ndiag]+1)
            x[:, :ndiag] = (x[:, :ndiag] - np.min(x[:, :ndiag], axis=0)) / np.ptp(x[:, :ndiag], axis=0)
            x[np.isnan(x)] = 0

        y_more = np.asarray(y_more)
        uid = np.asarray(uid_list)
        print('y_more, non-zero: ', (y_more.sum(0) > 0).sum(), ' ratio:', (y_more.sum(0) > 0).mean())
        if not use_behavior:
            self.col_name = (self.diag_code_vocab.feature_name(),
                             ['sex'],
                             ['age10-14', 'age15-19', 'age20-24'])
            self.FEATURE_NAME = np.asarray(sum(self.col_name, []))
            self.DIM_OF_CONFOUNDERS = len(self.FEATURE_NAME)
            print('Not use_behavior features, DIM_OF_CONFOUNDERS: ', self.DIM_OF_CONFOUNDERS)
        return x, y, uid, y_more
