import time ### temporary import for benchmarking
import argparse

from classifier import classifier
from utils import (get_accuracy,
                   config_loader
                   )

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    return parser.parse_args()

def main():
    args = argparser()
    config = config_loader(args.config)
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


if __name__ == "__main__":
    main()