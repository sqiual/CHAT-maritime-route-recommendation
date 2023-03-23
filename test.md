# CHAT-maritime-route-recommendation
This project is the official implementation of CHAT: Maritime Route Recommendation via Conditional Historical AIS Trajectory Diffusion

The project is conducted based on data provided by Danish Maritime Authority (DMA). We provided the processed data, or you can download the raw data from DMA (https://dma.dk/safety-at-sea/navigational-information/ais-data)


### Raw data preprocessing
**Skip this part if you choose to use our processed data.** For raw data preprocessing, please put the downloaded data into the data folder and kindly run the following scripts. (note the downloaded data file name should begin with "aisdk".)

````
python is_preprocessing_noint.py  
````

Further preprocess data to focus on the area defined through the "edit_hyper_parameters.json" file. You may change the latitude and longitude in the json file to include new areas
````
python edit_preprocessing.py  
````

This file only processes training data (the processed training data are in the data/imagedata/ folder). In this project, we include data from Jan-01-2022 to March-31-2022 for trajectory similarity measurement training
````
python traj2img.py 
````


### Prepare data for the project
**Skip this part if you choose to go from raw data.**  The processed training data for the trajectory similarity measurement module is provided at the data/imagedata/ folder (the output of the traj2img.py). The processed training and testing data for the route recommendation module is provided at the root directory named as "pred_train.hdf5", "pred_train_db.hdf5", "pred_test.hdf5", and "pred_train_db.hdf5".


### Trajectory Similarity Measurement Module Training
````
python trajsim_train.py
````


### Route Recommendation Module Training
We include data from April-01-2022 to May-31-2022 for route recommendation training
````
python recommendation_train.py
````

### Test
We include data from June-01-2022 to June-30-2022 for testing
````
python recommendation_test.py  
````

