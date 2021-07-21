import orbit_prediction.spacetrack_etl as etl
import orbit_prediction.build_training_data as training
import orbit_prediction.ml_model as ml

from sklearn.svm import LinearSVR
import sklearn.metrics as metrics
from sklearn.multioutput import MultiOutputRegressor

import pandas as pd

SPACETRACK_USERNAME=''
SPACETRACK_PASSWORD=''

def main():
    ## Importing ISS data
    spacetrack_client = etl.build_space_track_client( SPACETRACK_USERNAME,
                                                      SPACETRACK_PASSWORD )
    spacetrack_etl = etl.SpaceTrackETL(spacetrack_client)
    iss_orbit_data = spacetrack_etl.build_leo_df( norad_ids=['25544'],
                                                  last_n_days=30,
                                                  only_latest=None )
    
    ## Creating model and test data
    physics_model_predicted_orbits = training.predict_orbits( iss_orbit_data,
                                                              last_n_days=None,
                                                              n_pred_days=3 )
    physics_model_errors = training.calc_physics_error(physics_model_predicted_orbits)
    train_test_data = ml.build_train_test_sets(physics_model_errors, test_size=0.2)

    ## Attempt SVM fit
    lin_svm = LinearSVR(dual=False, loss='squared_epsilon_insensitive')
    svm_reg = MultiOutputRegressor(lin_svm)
    svm_reg.fit(train_test_data['X_train'], train_test_data['y_train'])

    lin_y_hat = svm_reg.predict(train_test_data['X_test'])
    rmse = metrics.mean_squared_error( train_test_data['y_test'], lin_y_hat,
                                       squared=False, multioutput='raw_values')

    print(rmse)

if __name__ == "__main__":
    main()
