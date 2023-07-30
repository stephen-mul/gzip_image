import config
import argparse
import os
import time
import csv
import pandas as pd
import numpy as np
from main import main
from utils import write_row

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='experiment_results')
    args = parser.parse_args()
    out_name = args.name

    if not os.path.exists('./results'):
            os.makedirs('./results')
    file = os.path.join('./results', out_name)

    if config.experiment_name=='TRAIN_SIZE':
        train_size_list = config.train_size_list
        headers = ['Correct', 'Incorrect', 'Training Set Size', 'Training Time']
        write_row(file, headers, 'w')
        for train_size in train_size_list:
            print(f'Testing with training set of size {train_size} and testing set of size ' 
                  f'{config.test_size}')
            start = time.time()
            hit, miss = main(train_size=train_size)
            stop = time.time()
            row = [hit, miss, train_size, stop-start]
            write_row(file, row)

    elif config.experiment_name=='KN':
        kn_list = config.kn_list
        headers = ['Correct', 'Incorrect', 'KN', 'Training Set Size', 'Training Time']
        write_row(file, headers)
        for kn in kn_list:
            print(f'Number of k nearest neighbours {kn}, training set of size {config.train_size} ' 
                    f'and testing set of size {config.test_size}')
            start = time.time()
            hit, miss = main(k=kn)
            stop = time.time()
            row = [hit, miss, kn, config.train_size, stop-start]
            write_row(file, row)

    else:
        print('Set config.experiment_name to a valid experiment type: TRAIN_SIZE , KN ')

