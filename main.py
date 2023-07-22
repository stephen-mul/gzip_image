import config
from classifier import classifier
from utils import get_accuracy

def main():

    if config.dataset =='MNIST':
        from dataloaders import mnist_loader
        train_loader = mnist_loader(batch=1, train=True)
        test_loader = mnist_loader(batch=1, train=False)

    training_set = train_loader.get_set(config.train_size)
    testing_set = test_loader.get_set(config.test_size)

    mnist_classifier = classifier(training_set, testing_set, k=config.k, mode='add', normalise_combined=False)
    test_labels = mnist_classifier.classify()

    #test_labels = classifier(training_set=training_set, testing_set=testing_set, k=config.k)

    get_accuracy(test_labels=test_labels)


if __name__ == "__main__":
    main()