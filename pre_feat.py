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
import torch.utils.data
from tqdm import tqdm
print = functools.partial(print, flush=True)


def load_icd9_2_ccs():
    """
     https://www.hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp
     Single Level CCS (ZIP file, 201 KB).
    :return:
    """
    df_name = pd.read_csv(r'data/ccs_dxlabel2015.csv', dtype=str)
    ccs2name = {}
    for index, row in tqdm(df_name.iterrows(), total=len(df_name)):
        ccs = row[0]
        name = row[1]
        ccs2name[ccs] = name

    print('len(ccs2name): ', len(ccs2name))
    pickle.dump(ccs2name, open('pickles/ccs2name.pkl', 'wb'))

    df = pd.read_csv(r'data/ccs_dxref2015.csv', header=0)
    icd2ccs = {}
    ccs2icd = defaultdict(set)
    for index, row in tqdm(df.iterrows(), total=len(df)):
        icd = row[0].strip(' \'\"\t\n')
        ccs = row[1].strip(' \'\"\t\n')
        ccs_des = row[2].strip(' \'\"\t\n')
        icd_des = row[3].strip(' \'\"\t\n')
        icd2ccs[icd] = (ccs, ccs_des, icd_des)
        ccs2icd[ccs].add(icd)

    print('len(icd2ccs): ', len(icd2ccs))
    print('len(ccs2icd): ', len(ccs2icd))
    pickle.dump(icd2ccs, open('pickles/icd2ccs.pkl', 'wb'))

    return icd2ccs, ccs2icd, ccs2name


def pre_df_2_dict(df, outfile):
    print('In pre_df_2_dict...')
    uid_records = defaultdict(list)
    print('total records: ', df.shape)
    print('Warning: Female encoded as 1')
    n_no_record = 0
    for index, row in tqdm(df.iterrows(), total=len(df)):
        myuid = row['myuid']

        age = row['age']
        sex = 1 if row['sex'] == 'F' else 0
        ddat = row['ddat']

        # Event: first suicide attempt after the most recent non-suicide related hospitalization
        # suicid attempt flag
        outcome = row['sa_ALL_ind']
        # t2e:

        record = [ddat, outcome, sex, age]

        dx_on_visit = set([])
        for cname in ['dx{}'.format(i) for i in range(1, 11)]:
            col = row[cname]
            if pd.isna(col) or (col == ''):
                continue
            dxs = [x.strip() for x in col.split(',') if (x.strip() != '')]
            for x in dxs:
                dx_on_visit.add(x)

        if len(dx_on_visit) == 0:
            n_no_record += 1
            print('row index', index, ' myuid:', myuid, 'has no dxs')

        record += list(dx_on_visit)

        uid_records[myuid].append(record)

    pickle.dump(uid_records, open(outfile, 'wb'))
    print("Done! Total patients: ", len(uid_records))
    print('There are {} rows have no dx records'.format(n_no_record))
    return uid_records


def pre_dict_to_triples_1st_neg(uid_records, enfunc):
    # use first 3 digits of ICD 9 or
    # use ccs codes?
    # compare their dimensions!
    print('In pre_dict_to_triples_1st_neg...')
    print('len(uid_records):', len(uid_records))
    triples = []
    for uid, records in tqdm(uid_records.items()):
        # records: [ddat, outcome, sex, age, codes * ]
        dxs = []
        i = -1
        ddat = outcome = sex = age = None
        for rec in records:
            i += 1
            ddat, outcome, sex, age = rec[:4]
            if outcome:
                # the first records is not positive, thus i >= 1
                break
            encodes = set([enfunc(x) for x in rec[4:]])
            dxs.append(list(encodes))

        if outcome:
            triples.append([uid,
                            [dxs, records[i-1][3], sex],
                            [1, (ddat - records[i-1][0]).days]
                            ])
        else:
            triples.append([uid,
                            [dxs, age, sex],
                            [0, 0]
                            ])

    print('len(triples):', len(triples))
    pickle.dump(triples, open('pickles/final_pats_1st_neg_triples.pkl', 'wb'))
    return triples


def pre_dict_to_triples_1st_sui(uid_records, enfunc):
    # for patients with 1st records as suicide attempt:
    # use all diagnosis codes, and the age at the first suicide attempt diagnosis
    # all patients have outcome 1
    print('In pre_dict_to_triples_1st_sui...')
    print('len(uid_records):', len(uid_records))
    triples = []
    for uid, records in tqdm(uid_records.items()):
        # records: [ddat, outcome, sex, age, codes * ]
        dxs = []
        i = -1
        ddat = outcome = sex = age = None
        first_sui_age = 9999
        for rec in records:
            i += 1
            ddat, outcome, sex, age = rec[:4]
            if outcome and age < first_sui_age:
                first_sui_age = age
            encodes = set([enfunc(x) for x in rec[4:]])
            dxs.append(list(encodes))
            # break

        triples.append([uid,
                        [dxs, first_sui_age, sex],
                        [1, 0]
                        ])

    print('len(triples):', len(triples))
    pickle.dump(triples, open('pickles/final_pats_1st_sui_triples.pkl', 'wb'))
    return triples


if __name__ == '__main__':
    icd2ccs, ccs2icd, ccs2name = load_icd9_2_ccs()

    def enfunc(x):
        a = icd2ccs.get(x, [])
        if a:
            return a[0] + '_' + a[1]
        else:
            return '0_INVALID_NO_DX'

    with open(r'data/final_pats_1st_neg.pkl', 'rb') as f:
        df_1st_neg = pickle.load(f)
        data_1st_neg = pre_df_2_dict(df_1st_neg, 'pickles/final_pats_1st_neg_dict.pkl')
        print(len(data_1st_neg))
        data_1st_neg_triples = pre_dict_to_triples_1st_neg(data_1st_neg, enfunc)

    with open(r'data/final_pats_1st_sui.pkl', 'rb') as f:
        df_1st_sui = pickle.load(f)
        data_1st_sui = pre_df_2_dict(df_1st_sui, 'pickles/final_pats_1st_sui_dict.pkl')
        print(len(data_1st_sui))
        data_1st_sui_triples = pre_dict_to_triples_1st_sui(data_1st_sui, enfunc)

    print('Done')
