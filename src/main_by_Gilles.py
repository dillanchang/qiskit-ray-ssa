import orbit_prediction.spacetrack_etl as etl
import orbit_prediction.build_training_data as training
import orbit_prediction.ml_model as ml
import sklearn.metrics as metrics
import pandas as pd
import numpy as np
import xlsxwriter
import pickle

from sklearn.svm import *
from sklearn.svm import LinearSVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from IPython.display import display, HTML
from sklearn import preprocessing
from sklearn import utils
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from numpy.random import randint


def rmse(C, kernel, degree, epsilon, gamma):
    train_test_data = pickle.load(open("train_test_data.pkl", "rb"))
    rbf_svm= SVR(C=C, kernel=kernel, epsilon=epsilon, gamma=gamma,
             tol=0.0000001, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
    svm_reg = MultiOutputRegressor(rbf_svm)
    print("*** now fitting the x_train and y_train data ***")
    svm_reg.fit(train_test_data['X_train'], train_test_data['y_train'])
    lin_y_hat = svm_reg.predict(train_test_data['X_test'])
    rmse = metrics.mean_squared_error(train_test_data['y_test'], lin_y_hat,
                                           squared=False, multioutput='raw_values')
    print("*** now returning the rmse values ***")
    return rmse

def main(vary_name, C, kernel, degree, epsilon, gamma, file_name):
    print(vary_name, C, kernel, degree, epsilon, gamma, file_name)
    kernel=kernel
    names = [vary_name,'error_r_x','error_r_y','error_r_z',        #name of each column error percent
                 'error_v_x','error_v_y','error_v_z']
    #We want to replace the following optional array parameters with if statements
    #that automatically assigns our defined array parameters to be n
    #n = [1, 10, 100, 1000, 10000]          #n exist in {C}
    #n = [1e-2, 1e-4, 1e-8, 1e-16, 1e-32]   #n exist in {gamma}
    #n = [2, 3, 4, 5, 6, 7, 8, 9, 10,       #n exist in {degree}
    #   13, 17, 19, 23, 29, 31]           
    #n = [0.1, 0.01, 0.001, 0.0001, 0.00001]#n exist in {epsilon}
    #n = [1, 10, 100, 1000, 10000]          #n exist in {sigma}
    C_vary = False
    degree_vary = False
    epsilon_vary = False
    gamma_vary = False

    if vary_name == 'C':
        n=C
        C_vary=True
    if vary_name == 'degree':
        n=degree
        degree_vary=True
    if vary_name == 'epsilon':
        n=epsilon
        epsilon_vary=True
    if vary_name == 'gamma':
        n=gamm
        gamma_vary=True

    #save the data into an excel file
    workbook = xlsxwriter.Workbook(file_name) #open existing excel file
    worksheet = workbook.add_worksheet()                 #create a worksheet
    
    #initialize the rows and columns of the table
    row = 0
    col = 0
    worksheet.write(row, col, "SVR( C="+str(C)+", kernel="+str(kernel)+ 
        ", degree="+str(degree)+", epsilon="+str(epsilon)+ 
        ", tol=0.0000001, shrinking=True, cache_size=200, verbose=False, max_iter=-1)")
    row = 1
    #place all names along the zero-th row
    for name in range(0,len(names)):
        worksheet.write(row, col, names[name])
        col += 1
    #place n_pred_days input into corresponding row and col
    col = 0
    row = 2

    for i in range(0,len(n)):
        worksheet.write(row, col, n[i])
        row += 1
    #place x and v error data into row and col
    row = 2
    for i in range(0,len(n)):
        col = 1
        r_m_s_e = np.array(rmse(C=n[i] if C_vary==True else C, kernel=kernel, 
            degree=n[i] if degree_vary==True else degree, 
            epsilon=n[i] if epsilon_vary==True else epsilon,
            gamma=n[i] if gamma_vary==True else gamma))
        for j in range(0,len(r_m_s_e)):
            worksheet.write(row, col, r_m_s_e[j])
            col += 1
        row += 1
    #close workbook
    workbook.close()
if __name__ == "__main__":
    main(vary_name='C', C=[1, 10, 100, 1000, 10000], 
        kernel='poly', degree=3, epsilon=0.1, gamma=None, file_name='src/rmse_data.xlsx')
    #vary_name is a string name of the varying parameter (e.g. 'C', 'degree', 'gamma')
    #kernel is a string (e.g. 'linear', 'poly', 'rbf'')
    #C is a number or 1d array containing varying Constants
    #degree is a number or 1d array containg varying degree values
    #epsilon is a number or 1d array of containing varying epsilon values
    #gamma is a number of 1d array containing varing gamma values

