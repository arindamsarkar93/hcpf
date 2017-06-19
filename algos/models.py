
# coding: utf-8

# In[1]:


import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.misc import logsumexp
from scipy.special import lambertw

# In[5]:


class poisson_response:
    
    def __init__(self):
        self.lam = 0.0
    def set_param(self,lam):
        self.lam = lam
    def get_param(self):
        return self.lam
    
    def mle_update(self,non_zero_indices,data,no_sample):
        length = len(non_zero_indices)
        sum_ele = 0.0
        for i in range(0,no_sample):
            rand_ind = np.random.randint(low=0,high=length)
            sum_ele += data[non_zero_indices[rand_ind][0],non_zero_indices[rand_ind][1]]
        self.lam = sum_ele/no_sample
    
    def expectation(self,response,a_ui,n_trunc):
        q = np.zeros(shape=n_trunc)
        log_a = np.log(a_ui)
        for i in range(1,n_trunc+1):
            q[i-1] = (-i*self.lam) + (response-i)*np.log(i) + i*log_a + i - 1
        norm = logsumexp(q)
        q = np.exp(q-norm)
        expec = 0.0
        for i in range(1,n_trunc+1):
            expec += i*q[i-1]
        del q
        return expec
    
    def update_param(self):    
        return


class ztp_response:
    
    def __init__(self):
        self.lam = 0.0
    def set_param(self,lam):
        self.lam = lam
    def get_param(self):
        return self.lam

    def mle_update(self,non_zero_indices,data,no_sample):
        length = len(non_zero_indices)
        sum_ele = 0.0
        for i in range(0,no_sample):
            rand_ind = np.random.randint(low=0,high=length)
            sum_ele += data[non_zero_indices[rand_ind][0],non_zero_indices[rand_ind][1]]
        mu = sum_ele/no_sample
        self.lam = lambertw(-np.exp(-mu)*mu) + mu
    def expectation(self,response,a_ui,n_trunc):
        return 