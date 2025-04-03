import csv
import yaml
import glob
import os
import numpy as np

### misc ###

def mode(lst):
    return max(set(lst), key=lst.count)

def get_accuracy(test_labels):
    hit = 0
    miss = 0
    for n in test_labels:
        truth, predicted = n
        if truth == mode(predicted):
            hit+=1
        else:
            miss += 1
    print(f'Correct: {hit} \n Incorrect: {miss}')
    return hit, miss

def normalise(array, mean=0.5, std=0.5):
    """
    Function to normalise an array
    Args:
        array (np.array): Array to normalise
        mean (float): Mean to normalise to. Defaults to 0.5
        std (float): Standard deviation to normalise to. Defaults to 0.5
    Return:
        np.array: Normalised array
    """
    return (array-mean)/(std)

def write_row(csv_path, row, mode='a'):
    with open(csv_path, mode) as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(row)


### config ###

def config_loader(config_path: str) -> dict:
    """
    Function to load a config file
    Args:
        config_path (str): Path to the config file
    Returns:
        dict: Dictionary containing the config
    """
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

### results ###

def average_results(results_dir: str) -> tuple:
    """
    Function to average results from a directory of results
    Args:
        results_dir (str): Directory containing the results files
    Returns:
        tuple(list, list, list): Lists of average accuracies, standard deviations, time per image, and n_trains
    """

    ### list dirs in results_dir
    seeds = os.listdir(results_dir)
    seed_dirs = [os.path.join(results_dir, seed) for seed in seeds if os.path.isdir(os.path.join(results_dir, seed))]
    print("Seed directories:")
    print(seed_dirs)
    
    ### get n_trains from one seed dir
    n_trains = os.listdir(seed_dirs[0])
    avg_n_train_acc = []
    std_n_train_acc = []
    avg_n_train_time_per_image = []
    for n_train in n_trains:
        seed_results = []
        seed_times = []
        for seed_dir in seed_dirs:
            n_train_dir = os.path.join(seed_dir, n_train)
            result_files = glob.glob(os.path.join(n_train_dir, 'results.yaml'))
            if result_files:
                with open(result_files[0], 'r') as f:
                    results = yaml.load(f, Loader=yaml.FullLoader)
                    print(f"Results for seed {seed_dir} and n_train {n_train}:")
                    seed_results.append(results['accuracy'])
                    seed_times.append(results['time_per_image'])
        
        avg_accuracy = np.mean(seed_results)
        std_accuracy = np.std(seed_results)
        avg_time_per_image = np.mean(seed_times)
        avg_n_train_acc.append(avg_accuracy)
        std_n_train_acc.append(std_accuracy)
        avg_n_train_time_per_image.append(avg_time_per_image)
        print(f"Average accuracy for n_train {n_train}: {avg_accuracy}")
        print(f"Standard deviation of accuracy for n_train {n_train}: {std_accuracy}")

    return avg_n_train_acc, std_n_train_acc, avg_n_train_time_per_image, n_trains
