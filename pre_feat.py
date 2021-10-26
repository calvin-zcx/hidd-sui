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


def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    parser.add_argument('--encode', choices=['ccssingle', 'ccsmultiple'], default='ccssingle')
    args = parser.parse_args()
    return args


def load_icd9_2_ccssingle():
    """
     https://www.hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp
     Single Level CCS (ZIP file, 201 KB).
    :return:
    """
    df_name = pd.read_csv(r'data/ccs_single/ccs_dxlabel2015.csv', dtype=str)
    ccs2name = {}
    for index, row in tqdm(df_name.iterrows(), total=len(df_name)):
        ccs = row[0]
        name = row[1]
        ccs2name[ccs] = name

    print('len(ccs2name): ', len(ccs2name))
    pickle.dump(ccs2name, open('pickles/ccs2name.pkl', 'wb'))

    df = pd.read_csv(r'data/ccs_single/ccs_dxref2015.csv', header=0)
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


def load_icd9_2_ccsmultiple():
    """
     https://www.hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp
     Multi-Level CCS (ZIP file, 104 KB).
    :return:
    """
    df_name = pd.read_csv(r'data/ccs_multi/dxmlabel-13.csv', dtype=str)
    ccs2name = {}
    for index, row in tqdm(df_name.iterrows(), total=len(df_name)):
        ccs = row[0].strip(' \'\"\t\n')
        name = row[1].strip(' \'\"\t\n')
        ccs2name[ccs] = name

    print('len(ccs2name): ', len(ccs2name))
    pickle.dump(ccs2name, open('pickles/ccsmulti2name.pkl', 'wb'))

    df = pd.read_csv(r'data/ccs_multi/ccs_multi_dx_tool_2015.csv', header=0)
    icd2ccs = {}
    ccs2icd = defaultdict(set)
    for index, row in tqdm(df.iterrows(), total=len(df)):
        icd = row[0].strip(' \'\"\t\n')
        ccsl3 = row[5].strip(' \'\"\t\n')
        ccsl3_des = row[6].strip(' \'\"\t\n')
        ccsl2 = row[3].strip(' \'\"\t\n')
        ccsl2_des = row[4].strip(' \'\"\t\n')
        ccsl1 = row[1].strip(' \'\"\t\n')
        ccsl1_des = row[2].strip(' \'\"\t\n')
        if ccsl3:
            ccs = ccsl3
            ccs_des = ccsl3_des
        elif ccsl2:
            ccs = ccsl2
            ccs_des = ccsl2_des
        else:
            ccs = ccsl1
            ccs_des = ccsl1_des

        icd2ccs[icd] = (ccs, ccs_des)
        ccs2icd[ccs].add(icd)

    print('len(icd2ccs): ', len(icd2ccs))
    print('len(ccs2icd): ', len(ccs2icd))
    pickle.dump(icd2ccs, open('pickles/icd2ccsmulti.pkl', 'wb'))

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
            # if len(dxs) > 1:
            #     print(dxs)
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


def pre_dict_to_triples_1st_neg(uid_records, enfunc, exclude1visit=False):
    # use first 3 digits of ICD 9 or
    # use ccs codes?
    # compare their dimensions!
    print('In pre_dict_to_triples_1st_neg...')
    print('len(uid_records):', len(uid_records))
    triples = []
    n_1visit = 0
    n_pos = 0
    n_neg = 0
    for uid, records in tqdm(uid_records.items()):
        # records: [ddat, outcome, sex, age, codes * ]
        dxs = []

        ddat = outcome = sex = age = None
        if len(records) == 1:
            n_1visit += 1
            if exclude1visit:
                continue
        i = 0
        for rec in records:
            i += 1
            ddat, outcome, sex, age = rec[:4]
            # if outcome:
            #     # the first records is not positive, thus i >= 1
            #     break
            encodes = set([enfunc(x) for x in rec[4:]])
            dxs.append(list(encodes))
            if outcome:
                # the first records is not positive, thus i >= 1
                # 2021-10-25 include last visit with suicide attempt 1
                break

        if outcome:
            # 2021-10-25
            # last sequence is visit with sui attempt, not use last when prediction
            # add two behaviour features: number of visit, and the time between first and last non-sui visits
            # positive: 1, negative with two or more visits: 0, negative with 1 visit: -1
            triples.append([uid,
                            [dxs, records[-2][3], sex, len(dxs)-1, (records[-2][0] - records[0][0]).days],
                            [1, (records[-1][0] - records[0][0]).days]
                            ])
            n_pos += 1
        else:
            triples.append([uid,
                            [dxs, age, sex, len(dxs), (records[-1][0] - records[0][0]).days],
                            [2 if len(dxs) == 1 else 0, (records[-1][0] - records[0][0]).days]
                            ])  # use all for prediction
            n_neg += 1

    print('len(triples):', len(triples))
    print('n_1visit:', n_1visit, 'n_pos:', n_pos, 'n_neg:', n_neg)
    if exclude1visit:
        foname = 'pickles/final_pats_1st_neg_triples_exclude1visit_before20150930.pkl'
    else:
        foname = 'pickles/final_pats_1st_neg_triples_before20150930.pkl'

    pickle.dump(triples, open(foname, 'wb'))
    print('Dump to {} done!'.format(foname))

    return triples


def pre_dict_to_triples_1st_sui(uid_records, enfunc):
    # for patients with 1st records as suicide attempt:
    # use only first sui diagnosis codes, and the age at the first suicide attempt diagnosis
    # all patients have outcome 1
    print('In pre_dict_to_triples_1st_sui...')
    print('len(uid_records):', len(uid_records))
    triples = []
    n_1visit = n_2morevisit = n_1sui =  n_2moresui = 0
    for uid, records in tqdm(uid_records.items()):
        # records: [ddat, outcome, sex, age, codes * ]
        dxs = []
        i = -1
        ddat = outcome = sex = age = None
        first_sui_age = 9999

        if len(records) == 1:
            n_1visit += 1
        else:
            n_2morevisit += 1

        _nsui = 0
        for rec in records:
            outcome = rec[1]
            if outcome:
                _nsui += 1

        if _nsui == 1:
            n_1sui += 1
        else:
            n_2moresui += 1

        for rec in records:  # only add first record
            i += 1
            ddat, outcome, sex, age = rec[:4]
            if outcome and age < first_sui_age:
                first_sui_age = age

            encodes = set([enfunc(x) for x in rec[4:]])
            dxs.append(list(encodes))
            break
        # 2021-10-25
        # first sequence is visit with sui attempt, only use first
        # add two behaviour features: number of visit, and the time between first and last non-sui visits
        # 3 for this type of data
        triples.append([uid,
                        [dxs, first_sui_age, sex, len(dxs)-1, 0],
                        [3, 0]
                        ])

    print('len(triples):', len(triples))
    print('n_1visit:', n_1visit, 'n_2morevisit:', n_2morevisit, 'n_1sui:', n_1sui, 'n_2moresui:', n_2moresui)
    foname = 'pickles/final_pats_1st_sui_triples_before20150930.pkl'
    pickle.dump(triples, open(foname, 'wb'))
    print('Dump {} done!'.format(foname))
    return triples

#
# def pre_dict_to_triples_1st_sui_type2(uid_records, enfunc):
#     # use first 3 digits of ICD 9 or
#     # use ccs codes?
#     # compare their dimensions!
#     print('In pre_dict_to_triples_1st_sui_type2...')
#     print('len(uid_records):', len(uid_records))
#     triples = []
#     n_10star = 0
#     n_100 = 0
#     n_101 = 0
#     for uid, records in tqdm(uid_records.items()):
#         # records: [ddat, outcome, sex, age, codes * ]
#         dxs = []
#         i = -1
#         ddat = outcome = sex = age = None
#         i_first_0 = None
#         for rec in records:
#             i+=1
#             ddat, outcome, sex, age = rec[:4]
#             if not outcome:
#                 i_first_0 = i
#                 break
#         if i_first_0 is None:
#             continue
#         else:
#             n_10star += 1
#
#         i = -1
#         records = records[i_first_0:]
#         for rec in records:
#             i += 1
#             ddat, outcome, sex, age = rec[:4]
#             if outcome:
#                 # the first records is not positive, thus i >= 1
#                 break
#             encodes = set([enfunc(x) for x in rec[4:]])
#             dxs.append(list(encodes))
#
#         if outcome:
#             triples.append([uid,
#                             [dxs, records[i-1][3], sex],
#                             [1, (ddat - records[i-1][0]).days]
#                             ])
#             n_101 += 1
#         else:
#             triples.append([uid,
#                             [dxs, age, sex],
#                             [0, 0]
#                             ])
#             n_100 += 1
#     print('len(triples):', len(triples))
#     print('n_10star:', n_10star, 'n_101:', n_101, 'n_100', n_100)
#     pickle.dump(triples, open('pickles/final_pats_1st_sui_triples.pkl', 'wb'))
#     return triples


if __name__ == '__main__':
    args = parse_args()
    if args.encode == 'ccssingle':
        print('Encoding: ccs single')
        icd2ccs, ccs2icd, ccs2name = load_icd9_2_ccssingle()
    elif args.encode == 'ccsmultiple':
        print('Encoding: ccs multiple')
        icd2ccs, ccs2icd, ccs2name = load_icd9_2_ccsmultiple()
    else:
        raise ValueError

    def enfunc(x):
        a = icd2ccs.get(x, [])
        if a:
            return a[0] + '_' + a[1]
        else:
            return '0_INVALID_NO_DX'

    with open(r'data/final_pats_1st_neg_before20150930.pkl', 'rb') as f:
        df_1st_neg = pickle.load(f)
        data_1st_neg = pre_df_2_dict(df_1st_neg, 'pickles/final_pats_1st_neg_dict_before20150930.pkl')
        print(len(data_1st_neg))
        # , exclude1visit=True  only use patients >= 2 records
        # =false, then original as wanwan, however, 1 visit are negative
        data_1st_neg_triples = pre_dict_to_triples_1st_neg(data_1st_neg, enfunc)  #, exclude1visit=True)

    with open(r'data/final_pats_1st_sui_before20150930.pkl', 'rb') as f:
        df_1st_sui = pickle.load(f)
        data_1st_sui = pre_df_2_dict(df_1st_sui, 'pickles/final_pats_1st_sui_dict_before20150930.pkl')
        print(len(data_1st_sui))
        data_1st_sui_triples = pre_dict_to_triples_1st_sui(data_1st_sui, enfunc)
        # data_1st_sui_triples = pre_dict_to_triples_1st_sui_type2(data_1st_sui, enfunc)

    a = []
    for k, v in data_1st_neg.items():
        a.append(len(v))
    n_of_visits = pd.DataFrame(a).value_counts()
    n_of_visits.to_csv('debug/no_of_visits_per_person_1st_neg_before20150930.csv')

    print('Done')
