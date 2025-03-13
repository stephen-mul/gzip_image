import configs.config as config
import argparse
import os
import time
import csv
import pandas as pd
import numpy as np
from main import main_old
from utils import (write_row,
                   config_loader)

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
        write_row(file, headers, 'w')
        for kn in kn_list:
            print(f'Number of k nearest neighbours {kn}, training set of size {config.train_size} ' 
                    f'and testing set of size {config.test_size}')
            start = time.time()
            hit, miss = main(k=kn)
            stop = time.time()
            row = [hit, miss, kn, config.train_size, stop-start]
            write_row(file, row)

    elif config.experiment_name=='FRAC_K':
         train_size_list = config.frac_k_train_size_list
         kn_list = [((train_size/10) -1) for train_size in train_size_list]
         headers = ['Correct', 'Incorrect', 'Training Set Size', 'KN', 'Training Time']
         write_row(file, headers, 'w')
         count = 0
         for train_size in train_size_list:
              print(f'Number of k nearest neighbours {kn_list[count]}, training set of size {train_size} ' 
                    f'and testing set of size {config.test_size}')
              start = time.time()
              hit, miss = main(train_size, config.test_size, kn_list[count])
              stop = time.time()
              row = [hit, miss, train_size, kn_list[count], stop-start]
              write_row(file, row)
              count+=1

    else:
        print('Set config.experiment_name to a valid experiment type: TRAIN_SIZE , KN , FRAC_K')

