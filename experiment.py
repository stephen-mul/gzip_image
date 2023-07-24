import config
from main import main

if __name__ == '__main__':
    train_size_list = config.train_size_list
    for train_size in train_size_list:
        print(f'Testing with training set of size {train_size} and testing set of size {config.test_size}')
        main(train_size=train_size)