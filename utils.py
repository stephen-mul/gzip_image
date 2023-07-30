import csv

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

def normalise(array, mean=0.5, std=0.5):
    return (array-mean)/(std)

def write_row(csv_path, row, mode='a'):
    with open(csv_path, mode) as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(row)