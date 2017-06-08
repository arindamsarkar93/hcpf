
# coding: utf-8

# In[ ]:

def temp(a):
    print a

def non_zero_entries(mat):   ## takes a 2 dimensional numpy array 
    indices = []
    for i in range(0,mat.shape[0]):
        for j in range(0,mat.shape[1]):
            if mat[i,j] > 0:
                indices.append((i,j))
    return indices


# In[ ]:

def load_data():
    X = np.loadtxt('./pre_processed_data/bibtex/X_train.txt',delimiter=',')
    #Y = np.loadtxt('./pre_processed_data/bibtex/Y_train.txt',delimiter=',')
    x_train_mask = np.loadtxt('./pre_processed_data/bibtex/x_train_mask.txt')
    #y_train_mask = np.loadtxt('./pre_processed_data/bibtex/y_train_mask.txt')
    x = X*x_train_mask
    #y = Y*y_train_mask
    return x
    


# In[ ]:



