# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 13:31:53 2023

@author: kvdhe
"""

import config
import tqdm
import gzip
import png

from classifier import classifier
from utils import get_accuracy


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

training_set = train_loader.get_set(config.train_size)
testing_set = test_loader.get_set(config.test_size)

####################

def write_png(image, n=256):
    w = png.Writer(n, 1, greyscale=False)
    w.write(image, [range(n)])
    
    

for (x1 , y1) in tqdm(testing_set):
    x1 = x1.numpy()
    Cx1 = len(gzip.compress(x1))