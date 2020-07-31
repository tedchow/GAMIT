"""
Created on Dec 29 2019,
By zhixiang
"""

import os
import csv
import numpy as np
import pandas as pd
import tensorflow as tf

from utils.util import z_score, z_inverse


def basic_hyperparams():
    return tf.contrib.training.HParams(
        # model parameters
        dataset='HK',
        epoch=50,
        batch_size=1,
        lr=1e-4,
        dr=0, #dropout_rate
        isfcl=0,

        num_cell=1,
        # encoder parameter
        filters=[64, 64],
        c4_hidden_size=0,

        # input
        len_his=6,
        len_his2=6,
        len_pre=6, # length of predicted time steps
        len_f=1, # length of predicted time steps
        len_scale=3,
        isgraph=1,
        # decay_epoch=10,
        decay_steps=1000,
        global_step=1000,
        decay_rate=0.1,
    )  


def update_config(args, config):
    config.isfcl = args.isfcl
    config.dataset = args.dataset
    config.epoch = args.epoch
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.dr = args.dr
    config.Ks = args.Ks
    config.act_func = args.act_func
    config.Kt = min(args.Kt, args.len_his2)

    config.num_cell = args.num_cell
    # encoder parameter
    config.filters = [args.filters1, args.filters2]
    config.c4_hidden_size = args.filters1

    # input
    config.len_his = args.len_his
    config.len_his2 = args.len_his2
    config.len_pre = args.len_pre
    config.len_f = args.len_f
    config.isgraph = args.isgraph
    config.global_step= args.global_step
    config.decay_rate= args.decay_rate
    config.sampling = args.sampling
    config.merge = args.merge
    config.rg = args.rg

    if config.isgraph == 1:
        config.Kt = 1
    return config


def normalization(train, val, test):
    '''
    Parameters
    ----------
    train, val, test: np.ndarray

    Returns
    ----------
    stats: dict, two keys: mean and std

    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original

    '''

    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]

    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)

    def normalize(x):
        return (x - mean) / std #(std+0.000000001)

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)
    # print(train_norm)
    return {'mean': mean, 'std': std}, train_norm, val_norm, test_norm


def search_data(sequence_length, ):
    """ 
    Parameters
    ----------
    sequence_length: int, length of all history data
    
    num_batch: int, the number of batches will be used for training
    
    label_start_idx: int the first index of predicting target
    
    len_pre: int, the number of points will be predict for each sample

    points_per_hour: int, number of points per hour

    Returns
    ----------
    list[]

    """
    if points_per_hour < 1:
        raise ValueError("points_per_hour should be greater than 0!")
    
    if label_start_idx + len_pre > sequence_length:
        return None

    x_idx = []

def shuffle_data(train_data):
    """
    Parameters
    ----------
    train_data: dict with keys ['inputs', labels]

    Returns
    ---------
    train_data: dict
    """
    new_train_data = {}
    # shuffle index
    shuffle_index = np.random.permutation(train_data['inputs'].shape[0])

    # shuffle train_data
    new_train_data['inputs'] = train_data['inputs'][shuffle_index]
    new_train_data['labels'] = train_data['labels'][shuffle_index]

    return new_train_data

def get_all_samples(data, len_his, len_his2, len_pre, len_f=1):
    """
    Parameters
    ----------
    data: np with [sequence_length, num_node]

    len_his: int, length of input historical data

    len_pre: int, length of future prediction

    Returns
    ----------
    all_samples_inputs: np, [?, len_his1, n_linKs, len_his2, len_f]

    all_samples_lables: np, [?, len_pre, n_linKs, len_his2, len_f]
    """
    all_samples_inputs = []
    all_samples_lables = []
    # num_instance = int(data.shape[0] / len_his2) - len_his - len_pre
    num_instance = int(data.shape[0] - (len_his + len_pre) * len_his2) 
    print('data shape:\t', data.shape)
    print('num_instance:\t', num_instance)

    data = data[:, :, :len_f]
    all_samples_inputs = np.zeros((num_instance, len_his, len_his2, data.shape[-2], data.shape[-1]))
    all_samples_lables = np.zeros((num_instance, len_pre, len_his2, data.shape[-2], data.shape[-1]))
    print('data shape:\t', data.shape)
    for i in range(num_instance):
        # all_samples_inputs[i][j] = data[i+j:i+j+len_his,:]
            # try:
            for j in range(len_his):
                all_samples_inputs[i][j] = data[i + j*len_his2 : i+(j+1)*len_his2]
            # all_samples_lables[i][0] = np.zeros_like(data[i + j*len_his2:i+(j+1)*len_his2])
            for k in range(len_pre):
                all_samples_lables[i][k] = data[i + (len_his+k)*len_his2:i+(len_his+k+1)*len_his2]
            # except:
            #     print('i%s-j%s-k%s' % (i, j, k))

    # all_samples_inputs = np.expand_dims(all_samples_inputs, axis=-1)
    # all_samples_lables = np.expand_dims(all_samples_lables, axis=-1)
    all_samples_inputs = np.transpose(all_samples_inputs, (0, 1, 3, 2, 4))
    all_samples_lables = np.transpose(all_samples_lables, (0, 1, 3, 2, 4))
    print('all_samples_inputs shape:\t', all_samples_inputs.shape)
    print('all_samples_lables shape:\t', all_samples_lables.shape)

    return all_samples_inputs, all_samples_lables

def get_st_all_samples(data, len_his, len_his2, len_pre, len_f=1):
    """
    Parameters
    ----------
    data: np with [sequence_length, num_node]

    len_his: int, length of input historical data

    len_pre: int, length of future prediction

    Returns
    ----------
    all_samples_inputs: np, [?, len_his1, n_linKs, len_his2, len_f]

    all_samples_lables: np, [?, len_pre, n_linKs, len_his2, len_f]
    """
    all_samples_inputs = []
    all_samples_lables = []
    # num_instance = int(data.shape[0] / len_his2) - len_his - len_pre
    num_instance = int(data.shape[0] - (len_his + len_pre) * len_his2 - 2*(Ks-1)) 
    print('data shape:\t', data.shape)
    print('num_instance:\t', num_instance)

    data = data[:, :, :len_f]
    all_samples_inputs = np.zeros((num_instance, len_his, len_his2+2*(Ks-1), data.shape[-2], data.shape[-1]))
    all_samples_lables = np.zeros((num_instance, len_pre, len_his2, data.shape[-2], data.shape[-1]))
    print('data shape:\t', data.shape)
    for i in range(num_instance):
        # all_samples_inputs[i][j] = data[i+j:i+j+len_his,:]
            # try:
            for j in range(len_his):
                all_samples_inputs[i][j] = data[i + j*len_his2 : i+(j+1)*len_his2+2*(Ks-1)]
            # all_samples_lables[i][0] = np.zeros_like(data[i + j*len_his2:i+(j+1)*len_his2])
            for k in range(len_pre):
                all_samples_lables[i][k] = data[i + (len_his+k)*len_his2:i+(len_his+k+1)*len_his2]
            # except:
            #     print('i%s-j%s-k%s' % (i, j, k))

    # all_samples_inputs = np.expand_dims(all_samples_inputs, axis=-1)
    # all_samples_lables = np.expand_dims(all_samples_lables, axis=-1)
    all_samples_inputs = np.transpose(all_samples_inputs, (0, 1, 3, 2, 4))
    all_samples_lables = np.transpose(all_samples_lables, (0, 1, 3, 2, 4))
    print('all_samples_inputs shape:\t', all_samples_inputs.shape)
    print('all_samples_lables shape:\t', all_samples_lables.shape)

    return all_samples_inputs, all_samples_lables

def data_train_valid_test(model_name='GcnLSTM', dataset='HK', len_his=6, len_his2=6, len_pre=6,\
    len_f=1, batch_size=32, train_r=0.7, valid_r=0.8, merge=0):
    split = 0

    data, adj = load_data(dataset)
    print('all data shape:\t', data.shape)
    mean = data[:int(data.shape[0] * 0.7), :, :len_f].mean(axis=0, keepdims=True)
    std = data[:int(data.shape[0] * 0.7), :, :len_f].std(axis=0, keepdims=True)
    print('std, mean', std.shape, mean.shape)
    data = z_score(data[:, :, :len_f], mean, std)
    all_samples_inputs, all_samples_lables = get_all_samples(data, len_his, len_his2, len_pre, len_f)

    split_line1 = int(all_samples_inputs.shape[0] * 0.7)
    split_line2 = int(all_samples_inputs.shape[0] * 0.8)


    if merge==1:
        split_line1 = split_line2

    train_batch_inputs, train_batch_lables = all_samples_inputs[:split_line1], \
                                             all_samples_lables[:split_line1]

    valid_batch_inputs, valid_batch_lables = all_samples_inputs[split_line1:split_line2], \
                                             all_samples_lables[split_line1:split_line2]

    test_batch_inputs, test_batch_lables = all_samples_inputs[split_line2:], \
                                           all_samples_lables[split_line2:]


    # stat, train_batch_inputs_norm, valid_batch_inputs_norm, test_batch_inputs_norm = \
    # normalization(train_batch_inputs, valid_batch_inputs, test_batch_inputs)
    # print('train_batch_inputs.shape:\t', train_batch_inputs)
    train_data = {'inputs': train_batch_inputs,
                  'labels': train_batch_lables}

    valid_data = {'inputs': valid_batch_inputs,
                  'labels': valid_batch_lables}

    test_data = {'inputs': test_batch_inputs, 
                 'labels': test_batch_lables}

    return train_data, valid_data, test_data, adj, std, mean


def load_data(dataset='PEMS08'):

    def get_adjacency_matrix(distance_df_filename, num_of_vertices):
        '''
        Parameters
        ----------
        distance_df_filename: str, path of the csv file contains edges information

        num_of_vertices: int, the number of vertices

        Returns
        ----------
        A: np.ndarray, adjacency matrix

        '''

        with open(distance_df_filename, 'r') as f:
            reader = csv.reader(f)
            header = f.__next__()
            edges = [(int(i[0]), int(i[1])) for i in reader]

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        for i, j in edges:
            A[i, j] = 1

        return A

    def load_PEMS_data(dataset='PEMS08', data_ty='flow'):
        pbase = '../data/'
        if not os.path.exists(pbase):
            print('data do not exist!!!!!!')
        data_path = pbase + dataset + '/'
        data_npz = np.load(data_path + '%s.npz' % dataset)
        adj = get_adjacency_matrix(data_path + 'distance.csv', num_of_vertices=data_npz['data'].shape[1])
        print('all data shape:\t', data_npz['data'].shape)
        data = data_npz['data']
        # print (adj)
        return data, adj

    if dataset in ['PEMS08', 'PEMS04']:
        print('======================== %s =================' % dataset)
        return load_PEMS_data(dataset)


        
def get_batch(data, batch_size,  dynamic_batch=False, shuffle=False):
    """
    data: dict with keys of ['inputs', 'labels']
    """
    len_data = data.shape[0]
    if shuffle:
        idx = np.arange(len_data)
        np.random.shuffle(idx)

    for start_idx in range(0, len_data, batch_size):
        end_idx = start_idx + batch_size
        if end_idx > len_data:
            if dynamic_batch:
                end_idx = len_data
            else:
                break
        if shuffle:
            slide = idx[start_idx:end_idx]
        else:
            slide = slice(start_idx, end_idx)
        # for key in data.keys():
        #     data[key] = data[key][slide]
        yield data_input[slide], data_label[slide]

def get_batch_input_dict(model, train_data, len_f=2):
    feed_dict = {
        model.phs['inputs']: train_data['inputs'],
        model.phs['labels']: train_data['labels'][:,:,:,:,:len_f],
        model.phs['scale_labels']: train_data['labels']}
    return feed_dict

def get_batch_input_lstm_dict(model, train_data):
    feed_dict = {
        model.phs['inputs']: train_data['inputs'],
        model.phs['labels']: train_data['labels']}
    return feed_dict

if __name__ == '__main__':
    load_data()