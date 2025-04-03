import time
import argparse
import os
import numpy as np
import yaml

from classifier import classifier
from utils import (get_accuracy,
                   config_loader
                   )

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--random_seed', type=int, default=0)
    return parser.parse_args()

def main():
    args = argparser()
    config = config_loader(args.config)
    
    ### create output directory
    output_dir = config['output_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ### Set random seed for reproducibility
    np.random.seed(args.random_seed)

    ### make subdirectory for given seed
    seed_dir = os.path.join(output_dir, str(args.random_seed))
    if not os.path.exists(seed_dir):
        os.makedirs(seed_dir)

    ### make subdirectory for given n_train
    n_train_dir = os.path.join(seed_dir, str(config['n_train']))
    if not os.path.exists(n_train_dir):
        os.makedirs(n_train_dir)

    ### copy config file to train directory
    config_file = os.path.join(n_train_dir, os.path.basename(args.config))
    if not os.path.exists(config_file):
        with open(args.config, 'r') as f:
            config_content = f.read()
        with open(config_file, 'w') as f:
            f.write(config_content)

    train_size = config['n_train']
    test_size = config['n_test']
    k = config['k']

    if config['dataset'] =='MNIST':
        from dataloaders import mnist_loader
        train_loader = mnist_loader(batch=1, train=True)
        test_loader = mnist_loader(batch=1, train=False)
    elif config['dataset'] =='CIFAR10':
        from dataloaders import cifar10_loader
        train_loader = cifar10_loader(batch=1, train=True)
        test_loader = cifar10_loader(batch=1, train=False)
    elif config['dataset'] =='FASHION':
        from dataloaders import fashion_mnist_loader
        train_loader = fashion_mnist_loader(batch=1, train=True)
        test_loader = fashion_mnist_loader(batch=1, train=False)

    training_set = train_loader.get_set(train_size)
    testing_set = test_loader.get_set(test_size)

    mnist_classifier = classifier(training_set, 
                                  testing_set, 
                                  k=k, 
                                  mode=config['mode'], 
                                  normalise_combined=config['normalise_combined'],
                                  compression_type=config['compression_type']
                                  )
    t_start = time.time()
    test_labels = mnist_classifier.classify()
    t_stop = time.time()
    print(f'Total classification time: {t_stop-t_start}')
    print(f'Time per image: {(t_stop-t_start)/config["n_test"]}')

    hit, miss = get_accuracy(test_labels=test_labels)
    # Return number of hit and misses if we are running an experiment
    if config['experiment_name'] != 'None':
        return hit, miss
    ### write out results to yaml file
    result_file = os.path.join(n_train_dir, 'results.yaml')
    with open(result_file, 'w') as f:
        yaml.dump({
            'hit': hit,
            'miss': miss,
            'accuracy': hit/(hit+miss),
            'time_per_image': (t_stop-t_start)/config['n_test'],
            'total_time': t_stop-t_start
        }, f)


if __name__ == "__main__":
    main()