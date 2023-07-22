def mode(lst):
    return max(set(lst), key=lst.count)

def get_accuracy(test_labels):
    hit = 0
    miss = 0
    for n in test_labels:
        truth, predicted = n
        if truth == mode(predicted):
            hit+=1
        else:
            miss += 1
    print(f'Correct: {hit} \n Incorrect: {miss}')
    return hit, miss