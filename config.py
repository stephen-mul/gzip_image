'''Basic config settings'''
dataset='MNIST'
train_size=200
test_size=100
k=5
mode='add'
normalise_combined=False
compression_type='gzip'

'''Experiment config settings'''
experiment_name='TRAIN_SIZE'
train_size_list = [500, 1000, 2000, 4000, 8000, 16000, 32000]
kn_list = [1, 2, 5, 10, 25, 50, 100]