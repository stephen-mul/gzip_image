import config
import argparse
import os
import time
import pandas as pd
import numpy as np
from main import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='experiment_results')
    args = parser.parse_args()

    if config.experiment_name=='TRAIN_SIZE':
        train_size_list = config.train_size_list
        headers = ['Correct', 'Incorrect', 'Training Set Size', 'Training Time']
        result_array = np.zeros((len(train_size_list), len(headers)))
        row_count = 0
        for train_size in train_size_list:
            start = time.time()
            print(f'Testing with training set of size {train_size} and testing set of size ' 
                  f'{config.test_size}')
            hit, miss = main(train_size=train_size)
            stop = time.time()
            result_array[row_count, 0] = hit
            result_array[row_count, 1] = miss
            result_array[row_count, 2] = train_size
            result_array[row_count, 3] = stop-start
            row_count += 1
        out_name = args.name
        if not os.path.exists('./results'):
            os.makedirs('./results')
        df = pd.DataFrame(result_array, columns=headers)
        df.to_csv(os.path.join('./results/', out_name))

    elif config.experiment_name=='KN':
        kn_list = config.kn_list
        headers = ['Correct', 'Incorrect', 'KN', 'Training Set Size', 'Training Time']
        result_array = np.zeros((len(kn_list), len(headers)))
        row_count=0
        for kn in kn_list:
            start = time.time()
            print(f'Number of k nearest neighbours {kn}, training set of size {config.train_size} ' 
                    f'and testing set of size {config.test_size}')
            hit, miss = main(k=kn)
            stop = time.time()
            result_array[row_count, 0] = hit
            result_array[row_count, 1] = miss
            result_array[row_count, 2] = kn
            result_array[row_count, 3] = config.train_size
            result_array[row_count, 5] = stop-start
            row_count += 1
        out_name = args.name
        if not os.path.exists('./results'):
            os.makedirs('./results')
        df = pd.DataFrame(result_array, columns=headers)
        df.to_csv(os.path.join('./results/', out_name))

    else:
        print('Set config.experiment_name to a valid experiment type: TRAIN_SIZE , KN ')

