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


def print_1_vs_more(df):
    p_nrecords = df['myuid'].value_counts()
    p_1 = [x for x, n in p_nrecords.items() if n == 1]
    p_1more = [x for x, n in p_nrecords.items() if n > 1]
    print('Shape{}, among {} pats: {} ({:.2f}%) one records, {} ({:.2f}%) one more records'.format(
        df.shape,
        p_nrecords.shape[0],
        len(p_1),
        len(p_1) / p_nrecords.shape[0] * 100.0,
        len(p_1more),
        len(p_1more) / p_nrecords.shape[0] * 100.0)
    )


def exclude():
    # print_hi('PyCharm')
    start_time = time.time()
    df = pd.read_csv('data/hidd_0517_ct_sa.csv', dtype=str, parse_dates=['dob', 'adat', "ddat", "fiscalyear"])
    df['age'] = df['age'].astype(int)
    df['sa_ALL_ind'] = df['sa_ALL_ind'].astype(int)
    print('Total:\n', df.shape)
    print_1_vs_more(df)

    print('Remove claims not in 2012-01-01 -- 2017-09-30:')
    df_exclude = df.loc[(df['fiscalyear'] >= pd.to_datetime('2012-01-01')) & (
            df['fiscalyear'] <= pd.to_datetime('2017-09-30')), :]
    print_1_vs_more(df_exclude)

    print('Keep commercial claims only')
    df_exclude = df_exclude.loc[df_exclude["ppayercode"].str.contains("G|F|S|T", regex=True), :]
    print_1_vs_more(df_exclude)

    print('Keep age 10-24 years old')
    df_exclude = df_exclude.loc[(df_exclude["age"] >= 10) & (df_exclude["age"] <= 24), :]
    print_1_vs_more(df_exclude)

    df_exclude = df_exclude.sort_values(by=['myuid', 'ddat'])

    df_exclude.to_csv('data/final_pats.csv')
    pickle.dump(df_exclude, open('data/final_pats.pkl', 'wb'))

    patient_records = defaultdict(list)
    for index, row in df_exclude.iterrows():
        myuid = row['myuid']
        flag = row['sa_ALL_ind']
        date = row['ddat']
        patient_records[myuid].append((date, flag))

    patient_1st_sui = set([])
    for key, val in patient_records.items():
        if val[0][1] == 1:
            patient_1st_sui.add(key)

    print('len(patient_1st_sui): ', len(patient_1st_sui))

    df_1st_sui = df_exclude.loc[df_exclude['myuid'].isin(patient_1st_sui), :]
    df_1st_sui['n_rows'] = df_1st_sui['myuid'].apply(lambda x: len(patient_records[x]))
    print_1_vs_more(df_1st_sui)
    df_1st_sui.to_csv('data/final_pats_1st_sui.csv')
    pickle.dump(df_1st_sui, open('data/final_pats_1st_sui.pkl', 'wb'))

    df_1st_neg = df_exclude.loc[~df_exclude['myuid'].isin(patient_1st_sui), :]
    df_1st_neg['n_rows'] = df_1st_neg['myuid'].apply(lambda x: len(patient_records[x]))
    print_1_vs_more(df_1st_neg)
    df_1st_neg.to_csv('data/final_pats_1st_neg.csv')
    pickle.dump(df_1st_neg, open('data/final_pats_1st_neg.pkl', 'wb'))

    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    # Counter({1: 837, 2: 403, 3: 82, 4: 38, 5: 24, 6: 13, 7: 3, 8: 2, 9: 2, 12: 2, 11: 1, 10: 1})
    # age_stats = df_exclude.loc[(df_exclude['fiscalyear'] >= pd.to_datetime('2014-01-01')) &
    #                            (df_exclude['fiscalyear'] <= pd.to_datetime('2015-12-31')), :].groupby('myuid').agg(
    #     {'age': ['min', 'max']})
    # idx = df_exclude['age'].isna()
    # for index, row in df_exclude.iterrows():
    #     myuid = row['myuid']
    #     if myuid in age_stats.index:
    #         age = age_stats.loc[myuid, ('age', 'min')]
    #         if (age >= 10) and (age <= 24):
    #             idx[index] = True
    #
    # df_exclude = df_exclude.loc[idx, :]
    # print_1_vs_more(df_exclude)


def load_pkl(dump=False):
    with open(r'data/final_pats.pkl', 'rb') as f:
        final_pats = pickle.load(f)
        print_1_vs_more(final_pats)
        if dump:
            final_pats.to_csv('data/final_pats.csv')

    with open(r'data/final_pats_1st_sui.pkl', 'rb') as f:
        final_pats_1st_sui = pickle.load(f)
        print_1_vs_more(final_pats_1st_sui)
        if dump:
            final_pats_1st_sui.to_csv('data/final_pats_1st_sui.csv')

    with open(r'data/final_pats_1st_neg.pkl', 'rb') as f:
        final_pats_1st_neg = pickle.load(f)
        print_1_vs_more(final_pats_1st_neg)
        if dump:
            final_pats_1st_neg.to_csv('data/final_pats_1st_neg.csv')

    return final_pats, final_pats_1st_sui, final_pats_1st_neg


if __name__ == '__main__':
    final_pats, final_pats_1st_sui, final_pats_1st_neg = load_pkl()
    print_1_vs_more(final_pats)
    print_1_vs_more(final_pats_1st_sui)
    print_1_vs_more(final_pats_1st_neg)