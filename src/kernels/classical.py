import numpy as np
 
def calc_lin_gram(X,Z):
	return np.matmul(X,np.transpose(Z))
