### script for producing plots ###
import argparse
import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='/home/stephen/local_workdir/personal/gzip_results/default')
    return parser.parse_args()

def plot_n_train_vs_acc(n_trains, 
                        accuracies,
                        std_accuracies=None,
                        output_dir=None,
                        ):
    """Plot accuracy vs n_train."""
    plt.figure(figsize=(10, 6))
    if std_accuracies is not None:
        plt.errorbar(n_trains, accuracies, yerr=std_accuracies, fmt='o', capsize=5, color='blue')
    else:
        plt.errorbar(n_trains, accuracies, yerr=0, fmt='o', capsize=5)
    plt.plot(n_trains, accuracies, marker='o', color='blue')
    plt.title('Accuracy vs Number of Training Samples')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Accuracy')
    #plt.xscale('log')
    plt.grid(True)
    plt.xticks(n_trains)
    plt.show()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'accuracy_vs_n_train.png'))
    plt.close()

def plot_n_train_vs_time(n_trains,
                        times,
                        output_dir=None,
                        ):
    """Plot time vs n_train."""
    plt.figure(figsize=(10, 6))
    plt.plot(n_trains, times, marker='o', color='blue')
    plt.title('Time per Image vs Number of Training Samples')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Time (seconds)')
    #plt.xscale('log')
    plt.grid(True)
    plt.xticks(n_trains)
    plt.show()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'time_vs_n_train.png'))
    plt.close()

def main():
    args = argparser()
    results_dir = args.results_dir

    n_trains = [100, 1000, 10000]

    ### results dir is grouped by seed and then by n_train
    ### we want to average over seeds for each n_train
    ### and then plot the results for each n_train

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

    ### plot results
    output_dir = os.path.join(results_dir, 'plots')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f'n_train: {n_trains}')
    print(f'avg_n_train_acc: {avg_n_train_acc}')
    print(f'std_n_train_acc: {std_n_train_acc}')
    print(f'avg_n_train_time_per_image: {avg_n_train_time_per_image}')
    ### order n_trains in ascending order and reorder accuracies and times accordingly
    n_trains = np.array(n_trains)
    avg_n_train_acc = np.array(avg_n_train_acc)
    avg_n_train_time_per_image = np.array(avg_n_train_time_per_image)
    sorted_indices = np.argsort(n_trains)
    n_trains = n_trains[sorted_indices]
    avg_n_train_acc = avg_n_train_acc[sorted_indices]
    avg_n_train_time_per_image = avg_n_train_time_per_image[sorted_indices]
    print(f'n_train: {n_trains}')
    print(f'avg_n_train_acc: {avg_n_train_acc}')
    print(f'avg_n_train_time_per_image: {avg_n_train_time_per_image}')

    plot_n_train_vs_acc(n_trains, 
                        avg_n_train_acc,
                        std_accuracies=std_n_train_acc,
                        output_dir=output_dir,
                        )
    
    plot_n_train_vs_time(n_trains,
                        avg_n_train_time_per_image,
                        output_dir=output_dir,
                        )




    

if __name__ == "__main__":
    main()