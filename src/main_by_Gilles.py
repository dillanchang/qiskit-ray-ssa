#Last edited by Gilles Djomani on August 11, 2021

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

#rmse_plots can generate (1) rmse values of some given SVR paraments and 
#(2) Actual Error vs Predicted Error plots of each SVR parameter
def rmse_plots(C, kernel, degree, epsilon, gamma):
    #pull the train and test data from the saved pickle file
    train_test_data = pickle.load(open("train_test_data.pkl", "rb"))
    #set the condictions for LinearSVR and any other SVR
    if kernel == 'linear':
        SVR_svm = LinearSVR(C=C, epsilon=epsilon, tol=0.000000001, loss="epsilon_insensitive",
            fit_intercept=True, intercept_scaling=1.0, dual=True, verbose=0, 
            random_state=None, max_iter=100)
    else:
        SVR_svm= SVR(C=C, kernel=kernel, epsilon=epsilon, gamma=gamma,
                 tol=0.000000001, shrinking=True, cache_size=7000, verbose=False, max_iter=100)

    #do a regressions giving the SVR parameters
    svm_reg = MultiOutputRegressor(SVR_svm)
    #fit the data
    svm_reg.fit(train_test_data['X_train'], train_test_data['y_train'])         
    lin_y_hat = svm_reg.predict(train_test_data['X_test'])                     #dimension: 22 x 6
    rmse = metrics.mean_squared_error(train_test_data['y_test'], lin_y_hat,    #dimension: 22 x 6
                                squared=False, multioutput='raw_values')

    #the code below creates Actual Error vs Predicted Error plots of some given parameters for a gridsearch
    #In order to generate the pltos, a folder named the kernel type (e.g. 'linear', 'rbf', etc.) most be first created in the 
    #directory where the file is contained. Next, the code below must be uncommented.
    
    '''
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
    '''
    return rmse

#rmse_store() stores rmse-component values of rmse from some given gridsearch, 
#into individual kernel and rmse-component based pickle files
def rmse_store(kernel):
    kernel = kernel

    #create gridsearch parameters
    vary1 = [10, 500, 1000, 2500, 5000, 7500]         #C
    vary2 = [3, 5, 8, 13, 20, 24]               #degree 
    vary3 = [1e+5, 1e-2, 1e-5, 1e-8, 1e-16]     #epsilon
    vary4 = [1e+5, 'scale', 1e-5, 1e-8, 1e-16]  #gamma

    rmse_r_x = []
    rmse_r_y = []
    rmse_r_z = []
    rmse_v_x = []
    rmse_v_y = []
    rmse_v_z = []

    #create plots from the gridsearch
    for i in range(0,len(vary1)):
        for j in range(0,len(vary2)):
            for k in range(0,len(vary3)):
                for l in range(0,len(vary4)):
                                    #kernel types: linear, rbf, poly, sigmoid
                    rmse_plots(C=vary1[i], kernel=kernel, degree=vary2[j], epsilon=vary3[k], gamma=vary4[l])
                    main = np.array(rmse_plots(C=vary1[i], kernel='linear', degree=vary2[j], epsilon=vary3[k], gamma=vary4[l]))
                    main = main.tolist()

                    rmse_r_x.append(main[0])
                    rmse_r_y.append(main[1])
                    rmse_r_z.append(main[2])
                    rmse_v_x.append(main[3])
                    rmse_v_y.append(main[4])
                    rmse_v_z.append(main[5])
        
    pickle.dump(rmse_r_x, open(str(kernel)+"_rmse_r_x.pkl", "wb"))
    pickle.dump(rmse_r_y, open(str(kernel)+"_rmse_r_y.pkl", "wb"))
    pickle.dump(rmse_r_z, open(str(kernel)+"_rmse_r_z.pkl", "wb"))
    pickle.dump(rmse_v_x, open(str(kernel)+"_rmse_v_x.pkl", "wb"))
    pickle.dump(rmse_v_y, open(str(kernel)+"_rmse_v_y.pkl", "wb"))
    pickle.dump(rmse_v_z, open(str(kernel)+"_rmse_v_z.pkl", "wb"))

#main() first pulls from #rmse_store(), which stores rmse-component values of rmse from some given gridsearch, 
#into individual kernel and rmse-component based pickle files
#It then outputs the minmun value from each rmse-component set
def main():
    kernels = ['linear', 'rbf', 'sigmoid'] #set you kernels here
                                           #
    for i in range(0, len(kernels)):
        rmse_store(kernel=kernels[i])
    for i in range(0, len(kernels)):
        rmse_r_x = pickle.load(open(str(kernels[i])+"_rmse_r_x.pkl", "rb"))
        rmse_r_y = pickle.load(open(str(kernels[i])+"_rmse_r_y.pkl", "rb"))
        rmse_r_z = pickle.load(open(str(kernels[i])+"_rmse_r_z.pkl", "rb"))
        rmse_v_x = pickle.load(open(str(kernels[i])+"_rmse_v_x.pkl", "rb"))
        rmse_v_y = pickle.load(open(str(kernels[i])+"_rmse_v_y.pkl", "rb"))
        rmse_v_z = pickle.load(open(str(kernels[i])+"_rmse_v_z.pkl", "rb"))
        print("\nThe min and max rmse values for kernel:", kernels[i])
        print("The min rmse_r_x value is", min(rmse_r_x), ".")
        print("The min rmse_r_y value is", min(rmse_r_y), ".")
        print("The min rmse_r_z value is", min(rmse_r_z), ".")
        print("The min rmse_v_x value is", min(rmse_v_x), ".")
        print("The min rmse_v_y value is", min(rmse_v_y), ".")
        print("The min rmse_v_z value is", min(rmse_v_z), ".")

#in main() one can set the kernels parameters
#in rmse_store() one can set the C, degree, epsilon, and gamma parameters
#in rmse_plots() one can futher tweak the SVR paramenters by varying LinearSVR() and SVR() parameters
if __name__ == "__main__":
    main()