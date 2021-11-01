import os
import shutil
import zipfile

import urllib.parse
import urllib.request

import torch
import torch.utils.data
from dataset import *
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import csv
from collections import Counter, defaultdict
import pandas as pd
from utils import check_and_mkdir


def shell_for_ml():
    fo = open('shell_lr.cmd', 'w')  # 'a'
    n = 0
    fo.write('mkdir output/log\n')
    for seed in range(0, 20):
        cmd = "python main.py --random_seed {} 2>&1 | tee output/log/lr_{}.log\n".format(seed, seed)
        fo.write(cmd)
        n += 1
    fo.close()
    print('In total ', n, ' commands')

# def shell_for_ml():
#     fo = open('shell_lr.cmd', 'w')  # 'a'
#     n = 0
#     for seed in range(0, 20):
#         cmd = "python main.py --random_seed {} 2>&1 | tee output/log/lr_{}.log\n".format(seed, seed)
#         fo.write(cmd)
#         n += 1
#     fo.close()
#     print('In total ', n, ' commands')


def results_summary(model='LR'):
    r9 = []
    r95 = []
    raux_9 = []
    raux_95 = []
    for seed in range(0, 20):
        df = pd.read_csv('output/test_results_{}r{}.csv'.format(model, seed))
        r9.append(df.loc[0, :].tolist())
        r95.append(df.loc[1, :].tolist())
        try:
            raux_9.append(df.loc[2, :].tolist())
            raux_95.append(df.loc[3, :].tolist())
        except:
            print('No raux_9/95')

    writer = pd.ExcelWriter('output/results_summary_{}.xlsx'.format(model), engine='xlsxwriter')
    pd_r9 = pd.DataFrame(r9, columns=df.columns).describe()
    pd_r9.to_excel(writer, sheet_name='r9')
    pd_r95 =pd.DataFrame(r95, columns=df.columns).describe()
    pd_r95.to_excel(writer, sheet_name='r95')
    print('Auc: {:.2f} ({:.2f})'.format(pd_r9.iloc[1,0], pd_r9.iloc[2,0]))
    print('90% Specificity\t95% Specificity')
    print('Sensitivity:\n{:.2f} ({:.2f})\t{:.2f} ({:.2f})'.format(pd_r9.iloc[1,3], pd_r9.iloc[2,3], pd_r95.iloc[1,3], pd_r95.iloc[2,3]))
    print('PPV:\n{:.3f} ({:.3f})\t{:.3f} ({:.3f})'.format(pd_r9.iloc[1,4], pd_r9.iloc[2,4], pd_r95.iloc[1,4], pd_r95.iloc[2,4]))

    try:
        pd.DataFrame(raux_9, columns=df.columns).describe().to_excel(writer, sheet_name='raux_9')
        pd.DataFrame(raux_95, columns=df.columns).describe().to_excel(writer, sheet_name='raux_95')
    except:
        print('No raux_9/95')
# Close the Pandas Excel writer and output the Excel file.
    writer.save()


if __name__ == '__main__':
    # shell_for_ml()
    # results_model_selection_for_ml(cohort_dir_name='save_cohort_all_loose', model='LR')
    results_summary('LR')
    print('Done')
