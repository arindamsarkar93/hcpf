
import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import gc

def non_zero_entries(mat):   ## takes a 2 dimensional numpy array 
    indices = []
    for i in range(0,mat.shape[0]):
        for j in range(0,mat.shape[1]):
            if mat[i,j] > 0:
                indices.append((i,j))
    return tuple(indices)


def num_non_zero(mat):   ## takes a 2 dimensional numpy array
    num = 0.0
    for i in range(0,mat.shape[0]):
        for j in range(0,mat.shape[1]):
            if mat[i,j] != 0:
                num += 1.0
    return num


def load_data():
    X = np.loadtxt('../data/movielens/X.txt',delimiter=' ')  #np.loadtxt('../data/bibtex/X_train.txt',delimiter=',')
    x_train_mask = np.loadtxt('../data/movielens/train_mask.txt')  #np.loadtxt('../data/bibtex/x_train_mask.txt')
    x_test_mask = np.loadtxt('../data/movielens/test_mask.txt')   #np.loadtxt('../data/bibtex/x_test_mask.txt')
    x = X*x_train_mask
    #y = Y*y_train_mask
    return X,x,x_test_mask
    

def ndcg_score(test_mask,X,result):
        
    ndcg = 0.0
    users = X.shape[0]
    items = X.shape[1]
    for i in range(0,users):
        result_sort_index = np.argsort(result[i])[::-1]
        data_sort_index = np.argsort(X[i])[::-1]
        score = 0.0
        norm = 0.0
        count1 = count2 = 1.0

        for j in range(0,items):
                if test_mask[i,data_sort_index[j]] == 1:
                    count1 += 1
                    norm += X[i,data_sort_index[j]]/np.log2(count1)
                if test_mask[i,result_sort_index[j]] == 1:
                    count2 += 1
                    score += X[i,result_sort_index[j]]/np.log2(count2)
        if norm != 0:
            ndcg += score/norm
        else:
            ndcg += 0
    ndcg /= users
    del result_sort_index
    del data_sort_index
    gc.collect()
    return ndcg


def mae(test_mask,X,result):

    count = 0.0
    error = 0.0
    users = X.shape[0]
    items = X.shape[1]
    for i in range(0,users):
        for j in range(0,items):
            if test_mask[i,j] == 1:
                count += 1
                error += abs(X[i,j]-result[i,j])

    return error/count


def check(param,theta_sample,beta_sample,test_mask,X,metric='ndcg'):
    
    result = np.zeros(shape=[X.shape[0],X.shape[1]])
    no_sample = theta_sample.shape[0]
    for i in range(0,no_sample):
        result = np.add(result,np.matmul(theta_sample[i],beta_sample[i]))
    result /= no_sample
    param.sample(result)

    del result
    
    if metric == 'mae':
        return mae(test_mask,X,param.sampled)
    elif metric == 'ndcg':
        return ndcg_score(test_mask,X,param.sampled)
    
    gc.collect()