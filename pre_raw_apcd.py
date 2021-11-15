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
    p_nrecords = df['internal_member_id'].value_counts()
    p_1 = [x for x, n in p_nrecords.items() if n == 1]
    p_1more = [x for x, n in p_nrecords.items() if n > 1]
    if p_nrecords.shape[0] >0 :
        print('Shape{}, among {} pats: {} ({:.2f}%) one records, {} ({:.2f}%) one more records'.format(
            df.shape,
            p_nrecords.shape[0],
            len(p_1),
            len(p_1) / p_nrecords.shape[0] * 100.0,
            len(p_1more),
            len(p_1more) / p_nrecords.shape[0] * 100.0)
        )
    else:
        print('Shape{}, among {} pats'.format(
            df.shape,
            p_nrecords.shape[0],
            )
        )


def exclude(dump=False):
    # print_hi('PyCharm')
    start_time = time.time()
    # , nrows=20000
    df = pd.read_csv('data/apcd/sa_young_adjust.csv', dtype=str, # nrows=200000,
                     parse_dates=['first_service_dt', 'eligibility_start_dt', 'eligibility_end_dt', 'birth_dt'])
    df['age_at_visit'] = df['age_at_visit'].astype(int)
    df['age_in_2014'] = df['age_in_2014'].astype(int)
    df['icd_version'] = df['icd_version'].astype(int)
    for i in range(1,7):
        df['cohort_{}'.format(i)] = df['cohort_{}'.format(i)].map({'FALSE':0, 'TRUE':1})
        df['idx{}'.format(i)] = df['idx{}'.format(i)].map({'FALSE':0, 'TRUE':1})

    print('Total:', df.shape)
    print_1_vs_more(df)

    df_exclude = df

    print('remove records with gender == U')
    df_exclude = df_exclude.loc[(df_exclude["gender"] == 'F') | (df_exclude["gender"] == 'M'), :]
    print_1_vs_more(df_exclude)

    # print('Remove insurance eligibility < 1 year patients')
    # df_exclude = df_exclude.loc[
    #              (df_exclude['eligibility_end_dt']-df_exclude['eligibility_start_dt']).apply(lambda x: x.days>=365), :]
    # print_1_vs_more(df_exclude)

    print('Sort records by ID and their date')
    df_exclude = df_exclude.sort_values(by=['internal_member_id', 'first_service_dt'])

    if dump:
        df_exclude.to_csv('data/apcd/final_pats_apcd.csv')
        pickle.dump(df_exclude, open('data/apcd/final_pats_apcd.pkl', 'wb'))

    patient_records = defaultdict(list)
    for index, row in df_exclude.iterrows():
        myuid = row['internal_member_id']
        flag = row['cohort_1']
        date = row['first_service_dt']
        eligibility_start_dt = row['eligibility_start_dt']
        eligibility_end_dt = row['eligibility_end_dt']
        patient_records[myuid].append((date, flag, eligibility_start_dt, eligibility_end_dt))

    patient_1st_sui = set([])
    patient_1st_NOTsui = set([])
    patient_1st_NOTsui_and_is_pos = set([])
    n_patient_not_in_windows = n_patient_in_windows = 0

    for key, val in patient_records.items():
        has_1_record_in_window = False
        for r in val:
            if (pd.to_datetime('2014-01-01') <= r[0] <= pd.to_datetime('2015-12-31')) and \
                    ((r[2] < pd.to_datetime('2014-01-01')) and (pd.to_datetime('2016-01-01') <= r[3])):
                # ((r[2] <= pd.to_datetime('2014-01-01')) and (pd.to_datetime('2015-12-31') <= r[3]))
                has_1_record_in_window = True
                break
        if has_1_record_in_window:
            n_patient_in_windows += 1
            if val[0][1] == 1:
                patient_1st_sui.add(key)
            else:
                patient_1st_NOTsui.add(key)
                for r in val:
                    if r[1] == 1:
                        patient_1st_NOTsui_and_is_pos.add(key)
        else:
            n_patient_not_in_windows += 1

    print('n_patient_in_windows: ', n_patient_in_windows, 'n_patient_not_in_windows: ', n_patient_not_in_windows)
    print('len(patient_1st_sui): ', len(patient_1st_sui))
    df_1st_sui = df_exclude.loc[df_exclude['internal_member_id'].isin(patient_1st_sui), :]
    df_1st_sui['n_rows'] = df_1st_sui['internal_member_id'].apply(lambda x: len(patient_records[x]))
    print_1_vs_more(df_1st_sui)
    if dump:
        df_1st_sui.to_csv('data/apcd/final_pats_1st_sui_apcd.csv')
        pickle.dump(df_1st_sui, open('data/apcd/final_pats_1st_sui_apcd.pkl', 'wb'))

    print('len(patient_1st_NOTsui): ', len(patient_1st_NOTsui))
    df_1st_neg = df_exclude.loc[df_exclude['internal_member_id'].isin(patient_1st_NOTsui), :]
    df_1st_neg['n_rows'] = df_1st_neg['internal_member_id'].apply(lambda x: len(patient_records[x]))
    print_1_vs_more(df_1st_neg)
    if dump:
        df_1st_neg.to_csv('data/apcd/final_pats_1st_neg_apcd.csv')
        pickle.dump(df_1st_neg, open('data/apcd/final_pats_1st_neg_apcd.pkl', 'wb'))

    print('len(patient_1st_NOTsui_and_is_pos): ', len(patient_1st_NOTsui_and_is_pos))
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


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
    exclude(dump=True)
    # exclude_with_recruit_window()
    # final_pats, final_pats_1st_sui, final_pats_1st_neg = load_pkl()
    # print_1_vs_more(final_pats)
    # print_1_vs_more(final_pats_1st_sui)
    # print_1_vs_more(final_pats_1st_neg)
    print('Done')