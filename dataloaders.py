import torch
import random
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

### custom transforms ###

class randomRotation90:
    """
    Randomly rotate the image by 0, 90, 180 or 270 degrees.
    
    Attributes:
        None
    
    Methods:
        __call__(self, x): Apply the random rotation to the input image.
    """
    def __call__(self, x):
        # randomly rotate the image by 0, 90, 180 or 270 degrees
        angles = [0, 90, 180, 270]
        angle = random.choice(angles)
        return TF.rotate(x, angle)

class mnist_loader:
    def __init__(self, 
                 batch=1, 
                 train=True,
                 rotation=False):
        if rotation:
            self.loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist_data',
                                                                download=True,
                                                                train=train,
                                                                transform=transforms.Compose([
                                                                    transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                                    transforms.Normalize((0.5), (0.5)), # normalize inputs
                                                                    randomRotation90() # randomly rotate the image
                                                                ])),
                                                batch_size=batch,
                                                shuffle=True)
        else:
            self.loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist_data',
                                                                download=True,
                                                                train=train,
                                                                transform=transforms.Compose([
                                                                    transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                                    transforms.Normalize((0.5), (0.5)) # normalize inputs
                                                                ])),
                                                batch_size=batch,
                                                shuffle=True)
    def get_set(self, n_samples):
        set = []
        count = 0
        for n in self.loader:
            n[0] = torch.squeeze(n[0])
            set.append(n)
            count += 1
            if count==n_samples:
                break
        return set

class cifar10_loader:
    def __init__(self, batch=1, train=True):
        self.loader = torch.utils.data.DataLoader(datasets.CIFAR10('../cifar10_data',
                                                                download=True,
                                                                train=train,
                                                                transform=transforms.Compose([
                                                                    transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) # normalize inputs
                                                                ])),
                                                batch_size=batch,
                                                shuffle=True)
    def get_set(self, n_samples):
        set = []
        count = 0
        for n in self.loader:
            n[0] = torch.squeeze(n[0])
            set.append(n)
            count += 1
            if count==n_samples:
                break
        return set
    
class fashion_mnist_loader:
    def __init__(self, batch=1, train=True):
        self.loader = torch.utils.data.DataLoader(datasets.FashionMNIST('../fashion_mnist_data',
                                                                download=True,
                                                                train=train,
                                                                transform=transforms.Compose([
                                                                    transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                                    transforms.Normalize((0.5), (0.5)) # normalize inputs
                                                                ])),
                                                batch_size=batch,
                                                shuffle=True)
    def get_set(self, n_samples):
        set = []
        count = 0
        for n in self.loader:
            n[0] = torch.squeeze(n[0])
            set.append(n)
            count += 1
            if count==n_samples:
                break
        return set