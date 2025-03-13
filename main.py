import config
from classifier import classifier
from utils import get_accuracy
import time ### temporary import for benchmarking

def main(train_size=config.train_size, test_size=config.test_size, k=config.k):

    if config.dataset =='MNIST':
        from dataloaders import mnist_loader
        train_loader = mnist_loader(batch=1, train=True)
        test_loader = mnist_loader(batch=1, train=False)
    elif config.dataset =='CIFAR10':
        from dataloaders import cifar10_loader
        train_loader = cifar10_loader(batch=1, train=True)
        test_loader = cifar10_loader(batch=1, train=False)
    elif config.dataset =='FASHION':
        from dataloaders import fashion_mnist_loader
        train_loader = fashion_mnist_loader(batch=1, train=True)
        test_loader = fashion_mnist_loader(batch=1, train=False)

    training_set = train_loader.get_set(train_size)
    testing_set = test_loader.get_set(test_size)

    mnist_classifier = classifier(training_set, testing_set, k=config.k, mode=config.mode, 
                                  normalise_combined=config.normalise_combined,
                                  compression_type=config.compression_type)
    t_start = time.time()
    test_labels = mnist_classifier.classify()
    t_stop = time.time()
    print(f'Total classification time: {t_stop-t_start}')
    print(f'Time per image: {(t_stop-t_start)/config.test_size}')

    hit, miss = get_accuracy(test_labels=test_labels)
    # Return number of hit and misses if we are running an experiment
    if config.experiment_name != 'None':
        return hit, miss


if __name__ == "__main__":
    main()