#This code creates Actual Error vs Predicted Error plots 
#of some given parameters for a gridsearch
#edited by Gilles Djomani on August 5, 2021

import orbit_prediction.spacetrack_etl as etl
import orbit_prediction.build_training_data as training
import orbit_prediction.ml_model as ml
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
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

def main(C, kernel, degree, epsilon, gamma):
    train_test_data = pickle.load(open("train_test_data.pkl", "rb"))  
    if kernel == 'linear':
        SVR_svm = LinearSVR(C=C, epsilon=epsilon, tol=0.00001, loss='epsilon_insensitive', 
            fit_intercept=True, intercept_scaling=1.0, dual=True, verbose=0, 
            random_state=None, max_iter=100)
    else:
        SVR_svm= SVR(C=C, kernel=kernel, epsilon=epsilon, gamma=gamma,
                 tol=0.00001, shrinking=True, cache_size=7000, verbose=False, max_iter=100)
    svm_reg = MultiOutputRegressor(SVR_svm)

    svm_reg.fit(train_test_data['X_train'], train_test_data['y_train'])         
    lin_y_hat = svm_reg.predict(train_test_data['X_test'])                      #22 x 6
    rmse = metrics.mean_squared_error(train_test_data['y_test'], lin_y_hat,     
                                squared=False, multioutput='raw_values')

    file_name="SVR(C="+str(C)+"_kernel="+str(kernel)+"_degree="+str(degree)+"_\u03B5="+str(epsilon)+"_\u03B3="+str(gamma)+")"

    x_axis_for_r_x = []
    y_axis_for_r_x = []
    x_axis_for_r_y = []
    y_axis_for_r_y = []
    x_axis_for_r_z = []
    y_axis_for_r_z = []

    y_axis_data = np.array(lin_y_hat)
    y_axis_data = y_axis_data.tolist()

    for j in range(0, len(y_axis_data)):
        y_axis_for_r_x.append((y_axis_data[j][0]/(2*6.378e6))*10000)
        y_axis_for_r_y.append((y_axis_data[j][1]/(2*6.378e6))*10000)
        y_axis_for_r_z.append((y_axis_data[j][2]/(2*6.378e6))*10000)

    for i in range(0,len(y_axis_data)):
        x_axis_for_r_x.append((np.array(train_test_data['y_test']['physics_err_r_x'])[i]/(2*6.378e6))*10000)
        x_axis_for_r_y.append((np.array(train_test_data['y_test']['physics_err_r_y'])[i]/(2*6.378e6))*10000)
        x_axis_for_r_z.append((np.array(train_test_data['y_test']['physics_err_r_z'])[i]/(2*6.378e6))*10000)

    max_ = 0 
    if max_ < max(x_axis_for_r_x):
        max_ = max(x_axis_for_r_x)
    if max_ < max(x_axis_for_r_x):
        max_ = max(y_axis_for_r_x)

    x_axis_for_dashed_line = np.linspace(-max_,max_,20)
    y_axis_for_dashed_line = x_axis_for_dashed_line

    fig, ax = plt.subplots()
    plt.title(file_name)
    
    plt.plot(x_axis_for_dashed_line, y_axis_for_dashed_line, '--', color='red')
    plt.plot(x_axis_for_r_x, y_axis_for_r_x, 'o', color='blue', label='r_x')
    plt.plot(x_axis_for_r_y, y_axis_for_r_y, 'o', color='purple', label='r_y')
    plt.plot(x_axis_for_r_z, y_axis_for_r_z, 'o', color='green', label='r_z')
    plt.xlabel("Actual Error (km)")
    plt.ylabel("Predicted Error (km)")
    plt.xlim([-max_, max_])
    plt.xlim([-max_, max_])
    ax.legend()
    plt.savefig(kernel+'/'+file_name+'.jpg')

    print("\n*** rmse data for ", file_name, "is listed below ***")
    print(rmse)
if __name__ == "__main__":
    #create gridsearch parameters
    vary1 = [10, 500,3000, 7500]         #C
    vary2 = [3, 8, 13, 24]               #degree 
    vary3 = [1e+5, 1e-2, 1e-5, 1e-8]     #epsilon
    vary4 = [1e+5, 'scale', 1e-5, 1e+8]  #gamma
    
    #create plots from the gridsearch
    for i in range(0,len(vary1)):
        for j in range(0,len(vary1)):
            for k in range(0,len(vary1)):
                for l in range(0,len(vary1)):
                    main(C=vary1[i], kernel='linear', degree=vary2[j], epsilon=vary3[k], gamma=vary4[l])
    
    #main(C=3000, kernel='linear', degree=24, epsilon=1e-8, gamma='scale')
