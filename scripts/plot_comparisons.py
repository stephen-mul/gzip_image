### script for comparing results of different configurations ###
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import glob
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import average_results

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_exp_0', default='/home/stephen/local_workdir/personal/gzip_results/cifar10_default')
    parser.add_argument('--results_exp_1', default='/home/stephen/local_workdir/personal/gzip_results/cifar10_concat')
    parser.add_argument('--output_dir', default='/home/stephen/local_workdir/personal/gzip_results/plots/cifar10_comparison')
    return parser.parse_args()

def order_results(avg,
                  std,
                  time,
                  n_trains,
                  ):
    """
    Reorder n_trains in ascending order and reorder the other lists accordingly.
    Args:
        avg (list): List of average accuracies.
        std (list): List of standard deviations.
        time (list): List of times per image.
        n_trains (list): List of number of training samples.
    Returns:
        tuple: Reordered lists of average accuracies, standard deviations, times per image, and number of training samples.
    """
    order = np.argsort(n_trains)
    avg = np.array(avg)[order]
    std = np.array(std)[order]
    time = np.array(time)[order]
    n_trains = np.array(n_trains)[order]
    return avg.tolist(), std.tolist(), time.tolist(), n_trains.tolist()

def compare_accuracies(avg_0,
                       std_0,
                       avg_1,
                       std_1,
                       n_trains_0,
                       n_trains_1,
                       output_dir=None,
                       ):
    """
    Compare accuracies of two experiments.
    Args:
        avg_0 (list): List of average accuracies for experiment 0.
        std_0 (list): List of standard deviations for experiment 0.
        avg_1 (list): List of average accuracies for experiment 1.
        std_1 (list): List of standard deviations for experiment 1.
        n_trains_0 (list): List of number of training samples for experiment 0.
        n_trains_1 (list): List of number of training samples for experiment 1.
        output_dir (str): Directory to save the plots.
    Returns:
        None
    """
    ### check if n_trains are the same, raise error if not
    if n_trains_0 != n_trains_1:
        print(f"n_trains_0: {n_trains_0}")
        print(f"n_trains_1: {n_trains_1}")
        raise ValueError("n_trains are not the same for both experiments.")
    ### plot accuracies
    plt.figure(figsize=(10, 6))
    plt.errorbar(n_trains_0, avg_0, yerr=std_0, fmt='o', capsize=5, label='Experiment 0')
    plt.errorbar(n_trains_1, avg_1, yerr=std_1, fmt='o', capsize=5, label='Experiment 1')
    plt.title('Accuracy vs Number of Training Samples')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.xticks(n_trains_0)
    plt.show()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'accuracy_vs_n_train.png'))
    plt.close()

def compare_times(time_0,
                  time_1,
                  n_trains_0,
                  n_trains_1,
                  output_dir=None,
                  ):
    """"
    Compare times of two experiments."
    Args:
        time_0 (list): List of times per image for experiment 0.
        time_1 (list): List of times per image for experiment 1.
        n_trains_0 (list): List of number of training samples for experiment 0.
        n_trains_1 (list): List of number of training samples for experiment 1.
        output_dir (str): Directory to save the plots."
    Returns:
        None
    """
    ### check if n_trains are the same, raise error if not
    if n_trains_0 != n_trains_1:
        print(f"n_trains_0: {n_trains_0}")
        print(f"n_trains_1: {n_trains_1}")
        raise ValueError("n_trains are not the same for both experiments.")
    ### plot times
    plt.figure(figsize=(10, 6))
    plt.plot(n_trains_0, time_0, marker='o', label='Experiment 0')
    plt.plot(n_trains_1, time_1, marker='o', label='Experiment 1')
    plt.title('Time per Image vs Number of Training Samples')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.xticks(n_trains_0)
    plt.show()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'time_vs_n_train.png'))
    plt.close()

def main():
    args = argparser()
    results_dir_exp_0 = args.results_exp_0
    results_dir_exp_1 = args.results_exp_1
    output_dir = args.output_dir
    ### create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ### get average results for each experiment
    avg_0, std_0, time_0, n_trains_0 = average_results(results_dir_exp_0)
    avg_1, std_1, time_1, n_trains_1 = average_results(results_dir_exp_1)
    ### reorder results
    avg_0, std_0, time_0, n_trains_0 = order_results(avg_0, std_0, time_0, n_trains_0)
    avg_1, std_1, time_1, n_trains_1 = order_results(avg_1, std_1, time_1, n_trains_1)
    ### plot accuracies
    compare_accuracies(avg_0, std_0, avg_1, std_1, n_trains_0, n_trains_1, output_dir=args.output_dir)
    ### plot times
    compare_times(time_0, time_1, n_trains_0, n_trains_1, output_dir=args.output_dir)



if __name__ == "__main__":
    main()