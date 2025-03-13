import gzip
import numpy as np
from tqdm import tqdm

from utils import normalise

class classifier:
    def __init__(self, training_set, testing_set, k, mode, normalise_combined, compression_type) -> None:
        self.training_set = training_set
        self.testing_set = testing_set
        self.k = k

        if mode=='add' and normalise_combined:
            self.combine = lambda x1, x2: normalise(x1+x2, 0.5, 0.5)
        elif mode=='add' and not normalise_combined:
            self.combine = lambda x1, x2: x1 + x2
        elif mode=='mult' and normalise_combined:
            self.combine = lambda x1, x2: normalise(x1*x2, 0.5, 0.5)
        elif mode=='mult' and not normalise_combined:
            self.combine = lambda x1, x2: x1*x2
        elif mode=='hadamard' and normalise_combined:
            self.combine = lambda x1, x2: normalise(np.multiply(x1, x2))
        elif mode=='hadamard' and not normalise_combined:
            self.combine = lambda x1, x2: np.multiply(x1, x2)

        if compression_type=='gzip':
            self.compress = lambda x: len(gzip.compress(x))

    def classify(self):
        test_labels = []
        for (x1 , y1) in tqdm(self.testing_set):
            x1 = x1.numpy()
            Cx1 = self.compress(x1)
            #distance_from_x1 = []
            distance_from_x1 = np.zeros(len(self.training_set))
            count = 0
            for (x2 , _) in self.training_set:
                x2 = x2.numpy()
                Cx2 = self.compress(x2)
                x1x2 = self.combine(x1, x2)
                Cx1x2 = self.compress(x1x2)
                ncd = ( Cx1x2 - min( Cx1 , Cx2 ))/max(Cx1 , Cx2 )
                #distance_from_x1.append(ncd)
                distance_from_x1[count] = ncd
                count+=1
            #sorted_idx = np.argsort(np.array(distance_from_x1))
            sorted_idx = np.argsort(distance_from_x1)
            #count += 1
            k_classes = []
            for n in np.arange(self.k):
                n_training = sorted_idx[n]
                _, label = self.training_set[n_training]
                k_classes.append(int(label))
            test_labels.append((y1, k_classes))
        return test_labels