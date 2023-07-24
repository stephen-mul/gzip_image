import config
import argparse
import os
import pandas as pd
import numpy as np
from main import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='experiment_results')
    args = parser.parse_args()
    if config.experiment_name=='TRAIN_SIZE':
        train_size_list = config.train_size_list
        headers = ['Correct', 'Incorrect', 'Training Set Size']
        result_array = np.zeros((len(train_size_list), len(headers)))
        row_count = 0
        for train_size in train_size_list:
            print(f'Testing with training set of size {train_size} and testing set of size' 
                  f'{config.test_size}')
            hit, miss = main(train_size=train_size)
            result_array[row_count, 0] = hit
            result_array[row_count, 1] = miss
            result_array[row_count, 2] = train_size
            row_count += 1
        out_name = args.name
        if not os.path.exists('./results'):
            os.makedirs('./results')
        df = pd.DataFrame(result_array, columns=headers)
        df.to_csv(os.path.join('./results/', out_name))

