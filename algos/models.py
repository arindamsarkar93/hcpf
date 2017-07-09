
import numpy as np
from scipy.misc import logsumexp
from scipy.special import lambertw

class hpf:
    def __init__(self,users,items):
        self.sampled = np.zeros(shape=[users,items])
        
    def sample(self,count):
        self.sampled = count

class poisson_response:
    
    def __init__(self,users,items,n_trunc):
        self.lam = 0.0
        self.sampled = np.zeros(shape=[users,items])
        self.q = np.zeros((users,items,n_trunc))
        self.en = np.zeros((users,items))
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
        self.q = q
        return expec
    
    def expectation_mat(self,x,a,n_trunc):
        log_a = np.log(a)
        for i in range(1,n_trunc):
            self.q[:,:,i] = (-i*self.lam) + (x-i)*np.log(i) + i*log_a + i - 1
        
        norm = logsumexp(self.q,axis=2)
        self.q = np.exp(self.q-norm[:,:,np.newaxis])
                
        for i in range(1,n_trunc):
            self.en += i*self.q[:,:,i]
        self.en = np.where(x==0,0.,self.en)    
            
        return self.en
    
    def sample(self,count):
        self.sampled = self.lam*count
                               

class invgauss_response:
    
    def __init__(self,users,items):
        self.mu = 0.0
        self.lam = 0.0
        self.sampled = np.zeros(shape=[users,items])
        
    def set_param(self,mu,lam):
        self.mu = mu
        self.lam = lam
        
    def get_param(self):
        return self.mu,self.lam

    def mle_update(self,non_zero_indices,data,no_sample):
        length = len(non_zero_indices)
        sum_ele = 0.0
        rand_ind = np.random.randint(low=0,high=length,size=no_sample)
        for i in range(0,no_sample):
            sum_ele += data[non_zero_indices[rand_ind[i]][0],non_zero_indices[rand_ind[i]][1]]
        self.mu = sum_ele/no_sample
        
        for i in range(0,no_sample):
            self.lam += (1.0/data[non_zero_indices[rand_ind[i]][0],non_zero_indices[rand_ind[i]][1]] - 1.0/self.mu)
        self.lam /= no_sample
        self.lam = 1.0/self.lam
        del rand_ind

    def expectation(self,response,a_ui,n_trunc):

        q = np.zeros(shape=n_trunc)
        log_a = np.log(a_ui)
        q[0] = self.lam*(1.0/self.mu - 0.5/response) + log_a
        for i in range(2,n_trunc+1):
            q[i-1] = self.lam*(i/self.mu - (i*i)/(2*response)) + i*log_a - (i-1)*np.log(i-1) + i - 2
        norm = logsumexp(q)
        q = np.exp(q-norm)
        expec = 0.0
        for i in range(1,n_trunc+1):
            expec += i*q[i-1]
        del q
        return expec
    
    def sample(self,count):
        
        for i in range(0,self.sampled.shape[0]):
            for j in range(0,self.sampled.shape[1]):
                self.sampled[i,j] = np.random.wald(self.mu*count[i,j],self.lam*count[i,j]*count[i,j])