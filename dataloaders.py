import torch
from torchvision import datasets, transforms

class mnist_loader:
    def __init__(self, batch=1, train=True):
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