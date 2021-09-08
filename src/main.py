import orbit_prediction.spacetrack_etl as etl
import orbit_prediction.ml_model as ml
import orbit_prediction.build_training_data as training

import kernels.quantum as q_kernel
import kernels.classical as c_kernel

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.svm import SVR

SPACETRACK_USERNAME='username@email.com'
SPACETRACK_PASSWORD='password'

N_PRED_DAYS      = 1
EARTH_RAD        = 6.378e6
MEAN_ORBIT_SPEED = 7800.
SECONDS_IN_DAY   = 60.*60.*24.

plt.rcParams.update({'font.size': 20})

def query_norm_X_data(n_data, X_data_raw):
    X_data       = np.zeros((n_data,13))
    X_data[:,0]  = X_data_raw['elapsed_seconds']/(N_PRED_DAYS*SECONDS_IN_DAY)
    X_data[:,1]  = X_data_raw['start_r_x']/(2.*EARTH_RAD)
    X_data[:,2]  = X_data_raw['start_r_y']/(2.*EARTH_RAD)
    X_data[:,3]  = X_data_raw['start_r_z']/(2.*EARTH_RAD)
    X_data[:,4]  = X_data_raw['start_v_x']/(MEAN_ORBIT_SPEED)
    X_data[:,5]  = X_data_raw['start_v_y']/(MEAN_ORBIT_SPEED)
    X_data[:,6]  = X_data_raw['start_v_z']/(MEAN_ORBIT_SPEED)
    X_data[:,7]  = X_data_raw['physics_pred_r_x']/(2.*EARTH_RAD)
    X_data[:,8]  = X_data_raw['physics_pred_r_y']/(2.*EARTH_RAD)
    X_data[:,9]  = X_data_raw['physics_pred_r_z']/(2.*EARTH_RAD)
    X_data[:,10] = X_data_raw['physics_pred_v_x']/(MEAN_ORBIT_SPEED)
    X_data[:,11] = X_data_raw['physics_pred_v_y']/(MEAN_ORBIT_SPEED)
    X_data[:,12] = X_data_raw['physics_pred_v_z']/(MEAN_ORBIT_SPEED)
    return X_data

def query_norm_Y_data(n_data, Y_data_raw):
    Y_data      = np.zeros((n_data,6))
    Y_data[:,0] = Y_data_raw['physics_err_r_x']/(2.*EARTH_RAD)
    Y_data[:,1] = Y_data_raw['physics_err_r_y']/(2.*EARTH_RAD)
    Y_data[:,2] = Y_data_raw['physics_err_r_z']/(2.*EARTH_RAD)
    Y_data[:,3] = Y_data_raw['physics_err_v_x']/(MEAN_ORBIT_SPEED)
    Y_data[:,4] = Y_data_raw['physics_err_v_y']/(MEAN_ORBIT_SPEED)
    Y_data[:,5] = Y_data_raw['physics_err_v_z']/(MEAN_ORBIT_SPEED)
    return Y_data

# Returns unit code normalized input and output data for SVM from train_test_data
def get_svm_input_output(data):
    num_train = data['X_train']['elapsed_seconds'].shape[0]
    num_test  = data['X_test']['elapsed_seconds'].shape[0]
    X_train = query_norm_X_data(num_train, data['X_train'])
    X_test  = query_norm_X_data(num_test,  data['X_test'])
    Y_train = query_norm_Y_data(num_train, data['y_train'])
    Y_test  = query_norm_Y_data(num_test,  data['y_test'])
    return X_train, X_test, Y_train, Y_test

def main():
    # ## Importing ISS data
    # spacetrack_client = etl.build_space_track_client( SPACETRACK_USERNAME,
    #                                                   SPACETRACK_PASSWORD )
    # spacetrack_etl = etl.SpaceTrackETL(spacetrack_client)
    # iss_orbit_data = spacetrack_etl.build_leo_df( norad_ids=['25544'],
    #                                               last_n_days=365,
    #                                               only_latest=None )
    # physics_model_predicted_orbits = training.predict_orbits( iss_orbit_data,
    #                                                           last_n_days=None,
    #                                                           n_pred_days=N_PRED_DAYS )
    # pickle.dump(physics_model_predicted_orbits, open("data/iss_data.pkl")
    # physics_model_errors = training.calc_physics_error(physics_model_predicted_orbits)
    # train_test_data      = ml.build_train_test_sets(physics_model_errors, test_size=0.25)
    # X_train, X_test, Y_train, Y_test = get_svm_input_output(train_test_data)
    # pickle.dump( X_train , open( "data/X_train.pkl", "wb" ) )
    # pickle.dump( X_test  , open( "data/X_test.pkl", "wb" )  )
    # pickle.dump( Y_train , open( "data/Y_train.pkl", "wb" ) )
    # pickle.dump( Y_test  , open( "data/Y_test.pkl", "wb" )  )

    # Loading cleaned train and test data
    X_train = pickle.load( open( "data/X_train.pkl", "rb" ) )
    X_test  = pickle.load( open( "data/X_test.pkl", "rb" )  )
    Y_train = pickle.load( open( "data/Y_train.pkl", "rb" ) )
    Y_test  = pickle.load( open( "data/Y_test.pkl", "rb" )  )

    # Calculating Gram matrix using quantum kernel
    X_train_gram = q_kernel.calc_gram_sym(X_train)
    np.save('./data/quantum/X_train_gram_1_rep_0_bit_scale_08.npy',X_train_gram)
    X_test_gram  = q_kernel.calc_gram(X_test,X_train)
    np.save('./data/quantum/X_test_gram_1_rep_0_bit_scale_08.npy',X_test_gram)

    # Calculating Gram matrix using linear kernel
    X_train_gram = c_kernel.calc_lin_gram(X_train,X_train)
    np.save('./data/classical/X_train_lin.npy',X_train_gram)
    X_test_gram  = c_kernel.calc_lin_gram(X_test,X_train)
    np.save('./data/classical/X_test_lin.npy',X_test_gram)

    # Fitting Gram matrix using support vector regression
    X_train_gram = np.load('./data/quantum/X_train_gram_1_rep_0_bit_scale_08.npy')
    X_test_gram  = np.load('./data/quantum/X_test_gram_1_rep_0_bit_scale_08.npy' )
    svrs     = []
    scales   = []
    gram_exp = 2.0
    N        = X_train_gram.shape[0]
    N_a      = N - 100
    for idx in range(0,6):
        x1 = np.power(X_train_gram[0:N_a,0:N_a],gram_exp)
        y1 = Y_train[0:N_a,idx]
        x2 = np.power(X_train_gram[N_a+1:N,0:N_a],gram_exp)
        y2 = Y_train[N_a+1:N,idx]
        svr = SVR(kernel='precomputed',C=1e-2,epsilon=1e-5)
        svr.fit(x1,y1)
        svrs.append(svr)
        Y_pred = svr.predict(x2)
        v = y2
        w = Y_pred
        m,b = np.polyfit(v,w,1)
        scales.append(m)

    # Plotting Final Results for X position
    idx = 0
    var_str = str(idx)
    Y_pred = svrs[idx].predict(np.power(X_test_gram[:,0:N_a],gram_exp))
    Y_pred = (1./scales[idx])*Y_pred
    rlim = np.max(np.abs(Y_test[0:Y_test.shape[0],idx]))
    plt.figure(figsize=(8,7))
    plot_r = np.arange(-rlim,rlim,0.001)
    v = Y_test[0:Y_test.shape[0],idx]
    w = Y_pred
    correlation_matrix = np.corrcoef(v,w)
    print(correlation_matrix[0,1])
    diff = w-v
    print(np.sqrt(np.mean(w*w))*2*EARTH_RAD/1000.)
    print(np.sqrt(np.mean(diff*diff))*2*EARTH_RAD/1000.)
    plt.scatter(v*2*EARTH_RAD/1000.,w*2*EARTH_RAD/1000.,s=80)
    plt.plot(plot_r*2*EARTH_RAD/1000.,plot_r*2*EARTH_RAD/1000.,'--',color='red',linewidth=3)
    plt.title("X Position")
    plt.xlabel("Actual Error (km)")
    plt.ylabel("Predicted Error (km)")
    plt.xlim(-700,700)
    plt.ylim(-700,700)
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()
