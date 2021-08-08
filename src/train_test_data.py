import orbit_prediction.spacetrack_etl as etl
import orbit_prediction.build_training_data as training
import orbit_prediction.ml_model as ml
import pickle

SPACETRACK_USERNAME='gdjomani@ibm.com'
SPACETRACK_PASSWORD='IBMInternship4G!'

## Importing ISS data
spacetrack_client = etl.build_space_track_client(SPACETRACK_USERNAME,
                                                 SPACETRACK_PASSWORD )
spacetrack_etl = etl.SpaceTrackETL(spacetrack_client)
iss_orbit_data = spacetrack_etl.build_leo_df( norad_ids=['25544'],
                                              last_n_days=365,
                                              only_latest=None )
## Creating model and test data
physics_model_predicted_orbits = training.predict_orbits(iss_orbit_data,
                                last_n_days=None, n_pred_days=1)
physics_model_errors = training.calc_physics_error(physics_model_predicted_orbits)
train_test_data = ml.build_train_test_sets(physics_model_errors, test_size=0.25)
#print("shape of train_test_data:", train_test_data.size)

empty_list = []
#Open the pickle file in 'wb' so that you can write and dump the empty variable
pickle.dump(train_test_data, open("train_test_data.pkl", "wb"))

print("*** DONE DUMPING ***")