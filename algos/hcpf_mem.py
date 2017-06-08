from __future__ import division
# coding: utf-8

# In[ ]:

#Imports
import math
import numpy as np

from math import exp
from math import factorial
from math import log
from math import sqrt
from math import pi
from math import gamma
from math import ceil

from scipy.special import digamma
from scipy.stats import poisson

from scipy import stats
from statsmodels.base.model import GenericLikelihoodModel
from scipy.stats import norm
from scipy.stats import invgauss

import matplotlib.pyplot as plt

import time
import random

import tensorflow as tf
from edward.models import Poisson,Gamma


# In[ ]:

#- - - -Maximum Likelihood estimation code- - - -

#For Normal Distribution
mu_def = 0;
sigma_def = 1;

def normal_pdf(x,mu=mu_def,sigma=sigma_def):
    return norm.pdf(x, loc=mu, scale=sigma);

class _normal(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, **kwds):
        if exog is None:
            exog = np.zeros_like(endog);
            
        super(_normal, self).__init__(endog, exog, **kwds)
        
    def nloglikeobs(self,params):
        mu = params[0];
        sigma = params[1];
        
        return -np.log(normal_pdf(self.endog, mu = mu, sigma = sigma));
    
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):

        if start_params == None:
            # Reasonable starting values (??)
            start_params = [0,1];
        return super(_normal, self).fit(start_params=start_params,
                                     maxiter=maxiter, maxfun=maxfun,
                                     **kwds)
    

mu_def = 1;
rate_def=1;
    
def inv_gaussian_pdf(x,mu=mu_def,rate=rate_def):
    #change to double-param form
    y = (x * mu**2)/rate;
    
    return invgauss.pdf(y, mu);

class _inv_gaussian(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, **kwds):
        if exog is None:
            exog = np.zeros_like(endog);
            
        super(_inv_gaussian, self).__init__(endog, exog, **kwds)
        
    def nloglikeobs(self,params):
        mu = params[0];
        rate = params[1];
        
        return -np.log(inv_gaussian_pdf(self.endog,mu=mu,rate=rate));
    
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):

        if start_params == None:
            # Reasonable starting values (??)
            start_params = [1,1];
        return super(_inv_gaussian, self).fit(start_params=start_params,
                                     maxiter=maxiter, maxfun=maxfun,
                                     **kwds)
    
#Write and check for other distributions 
#(at least Poisson, Gamma, Binomial)
#- - -- - - END- - - -   - - -   - - -  - - - -   


# In[ ]:

epsilon=0.0000001; #to avoid math errors wherever applicable!
#HCPF

#load data [todo]
X = np.loadtxt('./pre_processed_data/bibtex/X_train.txt',delimiter=',');
train_mask = np.loadtxt('./pre_processed_data/bibtex/x_train_mask.txt');
#X_masked = X*train_mask;
y = X*train_mask;

#random data

#actual data!
rows = len(X); # <- Anomaly causing!
cols = len(X[0]);
    
C_u = rows;
C_i = cols;

#fake data y -> responses
#reproducable results
#np.random.seed(42)
#y = np.random.normal(loc=0, scale=2, size=[C_u, C_i]);

# In[ ]:

#y_nm = list(); #non-missing entries
#y_m = list(); #missing entries
#y_train_idx_pair = list(); #use all
#y_train_idx_nm_pair = list(); #use all nm
y_test_idx_pair = list(); #use all nm

#missing/non-missing entries
for i in range(C_u):
    for j in range(C_i):
        
        #y_train_idx_pair.append([i,j]);
        
        #if(y[i][j]!=0):
        #    y_nm.append([i,j]);
        #    y_train_idx_nm_pair.append([i,j]);

        #if(y[i][j]==0):
            #y_m.append([i,j]);
    
        if(train_mask[i][j]==0):
            y_test_idx_pair.append([i,j]);

#train.. val, test split - - - - - - - - - -
y_test_idx_pair = tuple(y_test_idx_pair);

#- - - - - - - - - - -END- - - - - - - - - - -


# In[ ]:


#1% non missing entries as val, 20% for test
#num_val = int(np.floor(0.01 * len(y_nm)));
#num_test = int(np.floor(0.2 * len(y_nm)));

#y_val_idx_pair = list();
#y_test_idx_pair = list();

#assert(num_test>0);
#print num_test
#y_val_entry_idx = np.random.choice(range(0,len(y_nm)),num_val,replace=False); #random sampling without replacement

#for i in range(len(y_val_entry_idx)):
#    idx = y_val_entry_idx[i];
#    y_val_idx_pair.append(y_nm[idx]);

#remaining_idx = [idx_pair for idx_pair in y_nm if (idx_pair not in y_val_idx_pair)];

#np.random.seed(42);
#y_test_entry_idx = np.random.choice(range(0,len(y_nm)),num_test,replace=False); #random sampling without replacement
#np.random.seed(None);
#y_test_entry_idx = np.random.choice(range(0,len(remaining_idx)),num_test,replace=False);

#for i in range(len(y_test_entry_idx)):
#    idx = y_test_entry_idx[i];
#    y_test_idx_pair.append(y_nm[idx]);

#remaining_idx = [idx_pair for idx_pair in remaining_idx if (idx_pair not in y_test_idx_pair)];

#y_train_idx_nm_pair = remaining_idx;
#y_train_idx_pair = remaining_idx + y_m; #taining indices - remaining non-missing + missing


# In[ ]:

#Fixed Hyperparameters
K = 160;
rho2 = 0.1;
xi = 0.7;
rho1 = 0.01;
omega = 0.01;
omega_bar = 0.1;
tau = 10000;

zero_mat = np.zeros([C_u, C_i]);

num_non_zero_entries = np.sum(y!=0);#len(y_train_idx_nm_pair); #in training. 
num_tr_entries = C_i * C_u; #len(y_train_idx_pair);

num_missing_entries = num_tr_entries - num_non_zero_entries;

E_n_ui = num_non_zero_entries/num_tr_entries; #sparsity to begin with

print E_n_ui;

E_n = np.ones([C_u,C_i])*E_n_ui; #is this a valid initialization?

eta = rho2 * math.sqrt(E_n_ui/K);
zeta = omega_bar * math.sqrt(E_n_ui/K);

theta=0.0; #needs change [*] -> MLE
kappa=0.0; #needs change [*] -> MLE


# In[ ]:

#MLE driver/interface

def perform_mle(dist,data):
    if dist=='normal':
        model = _normal(data);
        results = model.fit()

        mu_mle, sigma_mle = results.params
        
        theta = mu_mle/sigma_mle**2;
        kappa = sigma_mle**2;
        
        return [theta,kappa];
    
    elif dist=='inv_gaussian':
        model = _inv_gaussian(data);
        results = model.fit()

        mu_mle, rate_mle = results.params
        
        theta = -rate_mle/(2*mu_mle**2);
        kappa = sqrt(rate_mle);
        
        return [theta,kappa];

#- - - -END - - - -


# In[ ]:

def dist_mean(dist,n_expected):
    if(dist=='inv_gaussian'):
        #we have theta, kappa
        k = kappa * n_expected;
        
        rate_params = k**2;
        mu_params = np.sqrt(-0.5 * rate_params/theta);

        #mean is mu_param
        return mu_params;
    
    if(dist=='normal'):
        mu_params = theta * n_expected * kappa;
        
        return mu_params;
        

def mae(result):
    error = 0.0
    count = 0.0
    #for i in range(0,C_u):
    #    for j in range(0,C_i):
    #        if [i,j] in y_test_idx_pair :
    #            error += abs(X[i][j]-result[i][j])
    #            count += 1

    for [i,j] in y_test_idx_pair :
        error += abs(X[i][j]-result[i][j])
        count += 1

    error /= count
    #error = math.sqrt(error)
    
    return error;
    
def check(dist):    
    
    q_s = Gamma(a_s,b_s)
    q_v = Gamma(a_v,b_v)
    
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    init.run()
    no_sample = 100

    s_sample = q_s.sample(no_sample).eval()
    v_sample = q_v.sample(no_sample).eval()
    
    n = np.zeros([C_u,C_i]);
    result = np.zeros([C_u,C_i]);
    n_expected = np.zeros([C_u,C_i]);

    for i in range(0,no_sample):
        n = np.add(n,np.matmul(s_sample[i],np.transpose(v_sample[i])))
    n_expected = n/no_sample; #mean of poisson is rate param. So this is fine.

    #sample response
    #distribution specific
    result = dist_mean(dist,n_expected);
    
    return mae(result)


# In[ ]:

#- - - - - - - call svi on data - - - - - - - -
#globals
N_tr = 10; # <- Should change
q_n = np.zeros([C_u, C_i, N_tr+1]); #need to keep these global; N_tr - 0->end
lambda_ = np.zeros([C_u, C_i]);

a_s=list();
b_s=list();
a_v=list();
b_v=list();

#----------------------------------------------

def log_partition_fn(dist,theta):
    if dist == 'normal':
        return (theta**2)/2;
    
    elif dist == 'gamma':
        return -log(-theta);
    
    elif dist == 'inv_gaussian':
        return -sqrt(-2*theta);
    
    elif dist == 'poisson':
        return exp(theta);
    
    elif dist == 'binomial':
        return log(1+exp(theta));
    
    elif dist == 'neg_binomial':
        return -log(1-exp(theta));
    
    elif dist == 'ztp':
        return log(exp(exp(theta)) - 1);
    
    
    
def h_const(dist, x, kappa):
    if dist == 'normal':
        h = (1/sqrt(2*pi*kappa)) * exp(-x**2/(2*kappa));
    
    elif dist == 'gamma':
        h = (x**(kappa-1))/gamma(kappa);
    
    elif dist == 'inv_gaussian':
        h = (kappa/sqrt(2 * pi * (x+epsilon)**3)) * exp(-kappa**2 / (2*x)); #note the use of epsilon!
    
    elif dist == 'poisson':
        h = (kappa**x)/factorial(x);
    
    elif dist == 'binomial':
        h = factorial(kappa)/(factorial(x) * factorial(x-kappa));
    
    elif dist == 'neg_binomial':
        h = factorial(x+kappa-1)/(factorial(x) * factorial(kappa-1));
    
    elif dist == 'ztp':
        h = 0.0;
        for j in range(kappa):
            h = h + (-1)**j * (kappa-j)**x * factorial(kappa)/(factorial(j)*factorial(kappa-j));
            
        h = h/factorial(x);
        
    return h;
    
def get_values_from_pair(data_y,index_pairs):
    result = list();
    result = [data_y[i][j] for i,j in index_pairs];
    
    return result;

def eval_edm(x,dist,theta,kappa):
    result = exp(x*theta - kappa*log_partition_fn(dist,theta))*h_const(dist,x,kappa);
    return result;

def calc_val_ll(theta, kappa, dist, y_val_idx_pair):
    #for all pairs
    ll_nm = 0;
    global lambda_;
    for idx in range(len(y_val_idx_pair)):
        u = y_val_idx_pair[idx][0];
        i = y_val_idx_pair[idx][1];
        
        marginal = 0;
        
        for n in range(1,N_tr+1): #0 -> N_tr (0 causes div by zero in h_const)
            marginal = marginal + eval_edm(y[u][i],dist,theta,n*kappa) * poisson.pmf(n,lambda_[u][i]);
            
        ll_nm = ll_nm + log(marginal+epsilon);
        
    return ll_nm;

#SVI for HCPF
def svi_hcpf(dist,y):
    print 'hcpf alloc started'
    global val_ll_list;
    val_ll_list = list();
    
    #initialize any other Hyperparams:
    
    #MLE for theta, kappa
    y_vec = y.flatten();
    #y_vec = get_values_from_pair(y,y_train_idx_pair);
    
    global theta, kappa;
    
    #theta, kappa = perform_mle(dist,y_vec);
    theta = 1.02922697078;
    kappa = 0.0285410730752;

    #print theta,kappa;

    print 'mle done'
    
    #init parameters [FIXED]
    a_r = np.zeros(C_u) + rho1 + K*eta;
    a_w = np.zeros(C_i) + omega + K*zeta;
    
    #init parameters [Updates for these, present]
    t_u = tau;
    t_i = tau;
    
    b_r = np.zeros(C_u) + (rho1/rho2);
    
    global a_s, b_s, a_v, b_v;
    
    a_s = np.zeros([C_u,K]) + eta;
    b_s = np.zeros([C_u,K]) + rho2;
    
    b_w = np.zeros(C_i) + omega/omega_bar;
    
    a_v = np.zeros([C_i,K]) + zeta;
    b_v = np.zeros([C_i,K]) + omega_bar;
    
    #init local variational parameters
    #q_n, lambda_ declared above
    phi = np.zeros(K);

    print 'alloc done'
    
    #Repeat till convergence
    cnt = 0; #placeholder convergence condition
    curr_idx = 0;

    global q_n,lambda_; #you're accessing global values. - -> relocate to original
    
    while(True):
        #Pick a sample uniformly from training portion
        #start = time.time();

        #tr_idx = np.random.choice(range(1,len(y_train_idx_pair)+1),1,replace=False)[0];
        #tr_idx = int(ceil(random.uniform(0.1,len(y_train_idx_pair)))); #faster!
        #tr_idx_pair = y_train_idx_pair[tr_idx-1];

        #end = time.time()
        #print 'random_choice',end-start;
        u = int(np.ceil(np.random.uniform(0.01, C_u)))-1;
        i = int(np.ceil(np.random.uniform(0.01, C_i)))-1;

        #rem = r_idx % C_u;
        #div = int(r_idx / C_u)-1;
        
        #if(div<0):
        #    div=0;

        #[u,i] = [tr_idx_pair[0],tr_idx_pair[1]];

        #compute local variational parameters
        #old_lambda_ = np.zeros([C_u,C_i]); #debugging purpose
        
        #lambda_ = np.zeros([C_u, C_i]);

        #start = time.time()
        #update lambda_ui
        for k in range(K):
            lambda_[u][i] = lambda_[u][i] + (a_s[u][k]*a_v[i][k])/(b_s[u][k]*b_v[i][k]);

        #end = time.time()
        #print 'lambda_',end-start;

        #update q_n_ui
        #distribution specific
        #need this for calculating E_n_ui

        #q_n_ui = calc_q_n_ui(dist,N_tr,lambda_,kappa);
        #Distribution specific calculation
        
        #calculating q_n[u][i] dist
        q_n_ui = np.zeros(N_tr+1); #keep 0th empty for now, for convenience -- not required

        #start = time.time()
        for n in range(1,N_tr+1): #value 'n' is being used
            q_n_ui[n] = exp(-kappa*n*log_partition_fn(dist,theta)) * h_const(dist, y[u][i], n*kappa) * (lambda_[u][i])**n/factorial(n);

     	#end = time.time()
        #print 'q_n',end-start;

        #^Normalize
        #start = time.time()
        #q_n_sum = 0;
        #for n in range(N_tr+1):
        q_n[u][i] = q_n_ui/np.sum(q_n_ui);
            #q_n_sum = q_n_sum + q_n[u][i][n];

        #end = time.time()
        #print "q_n_norm",end-start;

        #Calculate E[n_ui] [check]
        #E_n_ui = q_n_sum/N_tr; #Not sure if this is correct way to do
        #start = time.time()
        E_n_ui=0.0;
        for n in range(1,N_tr+1):
            E_n_ui = E_n_ui + (n)*(q_n[u][i][n]);
            
        E_n[u][i] = E_n_ui;

        #end = time.time()
        #print "e_n", end-start;

        #update phi_uik
        #calculating phi[u][i] dist
        
        #start = time.time()
        #for k in range(K):
        phi[:] = np.exp(digamma(a_s[u]) - np.log(b_s[u]) + digamma(a_v[i]) - np.log(b_v[i]));
        #^doubtful on form of above [to check]

        #end = time.time()
        #print "phi", end-start;

        #^Normalize
        
        #check if all 0
        if sum(phi)==0: #set uniform
           phi[:] = 1/K;
        
        phi = phi/sum(phi);

        #what to do with prop? [--Resolved--]
        #start = time.time()
        #compute global variational parameters [error prone!]
        b_r[u] = (1 - t_u**(-xi)) * b_r[u] + t_u**(-xi) * (rho1/rho2 + np.sum(a_s[u]/b_s[u]));
        
        for k in range(K):
            
            a_s[u][k] = (1 - t_u**(-xi)) * a_s[u][k] + t_u**(-xi) * (eta + C_i * E_n_ui * phi[k]);
            
            b_s[u][k] = (1 - t_u**(-xi)) * b_s[u][k] + t_u**(-xi) * (a_r[u]/b_r[u] + C_i * (a_v[i][k]/b_v[i][k]));
            
        b_w[i] = (1 - t_i**(-xi)) * b_w[i] + t_i**(-xi) * (omega/omega_bar +  np.sum(a_v[i]/b_v[i]));
        
        for k in range(K):
            a_v[i][k] = (1 - t_i**(-xi)) * a_v[i][k] + t_i**(-xi) * (zeta + C_u * E_n_ui * phi[k]);
            
            b_v[i][k] = (1 - t_i**(-xi)) * b_v[i][k] + t_i**(-xi) * (a_w[i]/b_w[i] + C_u * a_s[u][k]/b_s[u][k]);
            
        #end = time.time()
        #print "updates", end-start;

        #update learning rates
        t_u = t_u + 1;
        t_i = t_i + 1;
        
        #optional - update theta, kappa
        
        
        cnt = cnt + 1;
        #check for convergence - [Actual - validation log likelihood converges] todo
        
        #compute validation log likelihood
        #val_ll = calc_val_ll(theta, kappa, dist, y_val_idx_pair);
        #val_ll_list.append(val_ll);

        #mae
        #calculate MAE

        if cnt%100000==0:
            mae_val = check(dist);
            print mae_val;
            mae_list[curr_idx]=mae_val;
            curr_idx = curr_idx + 1;
        
        if(cnt == max_iter):
            break;


# In[ ]:

def sample_from_edm(dist, theta, kappa):
    if(dist=='normal'):
        sigma_2 = kappa;
        sigma = sqrt(sigma_2);
        mu = theta * sigma_2;
        
        return np.random.normal(loc=mu, scale=sigma, size=1);
    
    #todo for other distributions


# In[ ]:

def calc_test_ll(theta, kappa, dist, y_test_idx_pair):
    #for all pairs
    ll_nm = 0;
    ll_m = 0;
    global lambda_;
    for idx in range(len(y_test_idx_pair)):
        u = y_test_idx_pair[idx][0];
        i = y_test_idx_pair[idx][1];
        
        marginal = 0;
        
        for n in range(1,N_tr+1): #0 -> N_tr (0 causes div by zero in h_const)
            marginal = marginal + eval_edm(X[u][i],dist,theta,n*kappa) * poisson.pmf(n,lambda_[u][i]);
        
        ll_nm = ll_nm + marginal;
        
        ll_m = ll_m + log(poisson.pmf(0,lambda_[u][i]));
        
    return (0.2*(num_missing_entries)*ll_m)/len(y_test_idx_pair) + ll_nm;


# In[ ]:

# - - - - - -test - - - - -

#test using test-log-likelihood -- 

dist = 'normal';

#call svi
#Format: svi_hcpf(dist,y):

max_iter = C_u * C_i; #maximum iterations, to break early!

val_ll_list = list();
mae_list = np.zeros(max_iter);

svi_hcpf(dist,y);

print theta,kappa

#Reality Check
#sample and check
y_sampled = np.zeros([C_u, C_i]);

loss = 0;

#test log likelihood
print calc_test_ll(theta, kappa, dist, y_test_idx_pair);

"""
for u in range(C_u):
    for i in range(C_i):
        #sample count -> use variational approx. q_n[u][i]
        #can treat above dist. as multinoulli, and draw from it'
        #random.multinomial returns frequency, 1 is in atmost one entry. So argmax gets that
        #n_sample = np.random.multinomial(1,q_n[u][i],size=1)[0].argmax();
        E_n_ui = E_n[u][i];
        y_sampled[u][i] = sample_from_edm(dist,theta, E_n_ui*kappa);
        
        loss = loss + (y_sampled[u][i] - y[u][i])**2;
        
print 'rms_obtained', sqrt(loss/(C_u*C_i));
        

#check significance
random_y = np.random.normal(loc=0, scale=2, size=[C_u, C_i]);
loss_r = 0;

for u in range(C_u):
    for i in range(C_i):        
        loss_r = loss_r + (random_y[u][i] - y[u][i])**2;
        
print 'rms_random', sqrt(loss_r/(C_u*C_i));
"""

#plot val_ll

plt.plot(mae_list);
plt.show();


# In[ ]:

#check 

