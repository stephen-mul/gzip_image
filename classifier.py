import torch
import gzip
import numpy as np

from utils import normalise

def classifier(training_set, testing_set, k=5):

    test_labels = []
    for (x1 , y1) in testing_set:
        x1 = x1.numpy()
        Cx1 = len(gzip.compress(x1))
        distance_from_x1 = []
        count = 0
        for (x2 , _) in training_set:
            x2 = x2.numpy()
            Cx2 = len(gzip.compress(x2))
            x1x2 = x1 + x2
            #x1x2 = x1*x2
            Cx1x2 = len(gzip.compress(normalise(x1x2, 0.1307, 0.3081)))
            #Cx1x2 = len(gzip.compress(x1x2))
            ncd = ( Cx1x2 - min( Cx1 , Cx2 ))/max(Cx1 , Cx2 )
            distance_from_x1.append(ncd)
        sorted_idx = np.argsort(np.array(distance_from_x1))
        count += 1
        k_classes = []
        for n in np.arange(k):
            n_training = sorted_idx[n]
            _, label = training_set[n_training]
            k_classes.append(int(label))
        test_labels.append((y1, k_classes))
    return test_labels