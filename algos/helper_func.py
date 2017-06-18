
# coding: utf-8

# In[ ]:


import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# In[ ]:


def non_zero_entries(mat):   ## takes a 2 dimensional numpy array 
    indices = []
    for i in range(0,mat.shape[0]):
        for j in range(0,mat.shape[1]):
            if mat[i,j] > 0:
                indices.append((i,j))
    return tuple(indices)


# In[ ]:


def num_non_zero(mat):   ## takes a 2 dimensional numpy array
    num = 0.0
    for i in range(0,mat.shape[0]):
        for j in range(0,mat.shape[1]):
            if mat[i,j] != 0:
                num += 1.0
    return num


# In[ ]:


def load_data():
    X = np.loadtxt('../data/movielens/X.txt',delimiter=' ')
    x_train_mask = np.loadtxt('../data/movielens/train_mask.txt')
    x_test_mask = np.loadtxt('../data/movielens/test_mask.txt')
    x = X*x_train_mask
    #y = Y*y_train_mask
    return X,x,x_test_mask
    


# In[ ]:


def ndcg_score(users,items,train_mask,X,result):
        
    ndcg = 0
    for i in range(0,users):
        sort_index = np.argsort(result[i])[::-1]
        score = 0
        norm = 0
        count = 1
        
        for j in range(0,items):
                if train_mask[i,sort_index[j]] == 0 and X[i,sort_index[j]] == 1:
                    count += 1
                    norm += 1.0/np.log2(count)
                    score += 1.0/np.log2(j+2)
        if norm != 0:
            ndcg += score/norm
        else:
            ndcg += 0
    ndcg /= users
    return ndcg


# In[ ]:


def check(q_theta,q_beta,users,items,train_mask,X,no_sample=100,metric='ndcg'):
    
    q_theta = Gamma(gam_shp,gam_rte)
    q_beta = Gamma(lam_shp,lam_rte)
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    init.run()
    beta_sample = q_beta.sample(no_sample).eval()
    theta_sample = q_theta.sample(no_sample).eval()
    result = np.zeros([users,items])
    for i in range(0,no_sample):
        result = np.add(result,np.matmul(theta_sample[i],np.transpose(beta_sample[i])))
    result /= no_sample
    
    if metric == 'mae':
        return mae(users,items,train_mask,X,result)
    elif metric == 'ndcg':
        return ndcg_score(users,items,train_mask,X,result)

