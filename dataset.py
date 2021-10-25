# import logging
import numpy as np
import torch.utils.data
from vocab import *
from tqdm import tqdm

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

        for uid, patient_confounder, patient_outcome in tqdm(self.patient_list):
            self.outcome.append(patient_outcome)
            diag_visit, age, sex = patient_confounder
            self.diagnoses_visits.append(diag_visit)
            self.sexes.append(sex)
            self.ages.append(age)
            self.uid.append(uid)

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
        self.outcome = np.asarray(self.outcome)

        # feature dim: med_visit, diag_visit, age, sex, days
        self.DIM_OF_CONFOUNDERS = len(self.diag_code_vocab) + 4
        print('DIM_OF_CONFOUNDERS: ', self.DIM_OF_CONFOUNDERS)

        # feature name
        diag_col_name = self.diag_code_vocab.feature_name()
        col_name = (diag_col_name, ['sex'], ['age10-14', 'age15-19', 'age20-24'])
        self.FEATURE_NAME = np.asarray(sum(col_name, []))

    def _process_visits(self, visits, max_len_visit, vocab):
        res = np.zeros((max_len_visit, len(vocab)))
        for i, visit in enumerate(visits):
            res[i] = self._process_code(vocab, visit)
        # col_name = [vocab.id2name.get(x, '') for x in range(len(vocab))]
        return res  # , col_name

    def _process_code(self, vocab, codes):
        multi_hot = np.zeros((len(vocab, )), dtype='float')
        for code in codes:
            if code in vocab.code2id:
                multi_hot[vocab.code2id[code]] = 1
        return multi_hot

    def _process_ages(self):
        ages = np.zeros((len(self.ages), 3))
        for i, x in enumerate(self.ages):
            if 10 <= x <= 14:
                ages[i, 0] = 1
            elif 15 <= x <= 19:
                ages[i, 1] = 1
            elif 20 <= x <= 24:
                ages[i, 2] = 1
            else:
                print(i, 'row, wrong age within [10, 24]: ', x)
                raise ValueError
        return ages

    def __getitem__(self, index):
        # Problem: very sparse due to 1. padding a lots of 0, 2. original signals in high-dim.
        # paddedsequence for 1 and graph for 2?
        # should give new self._process_visits and self._process_visits
        # also add more demographics for confounder
        diag = self.diagnoses_visits[index]
        diag = self._process_visits(diag, self.diag_visit_max_length, self.diag_code_vocab)  # T_dx * D_dx

        sex = self.sexes[index]
        age = self.ages[index]
        # outcome = self.outcome[index][self.outcome_type]  # no time2event using in the matter rising
        outcome = self.outcome[index]
        confounder = (diag, sex, age)

        uid = self.uid[index]

        return confounder, outcome, uid

    def __len__(self):
        return len(self.diagnoses_visits)

    def _to_tensor(self):
        # 2021/10/25
        # for static, pandas dataframe-like learning
        # refer to flatten_data function
        pass
