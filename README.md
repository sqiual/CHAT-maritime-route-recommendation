# CHAT-maritime-route-recommendation
This project is the official implementation of CHAT: Maritime Route Recommendation via Conditional Historical AIS Trajectory Diffusion

The project is conducted based on data provided by Danish Maritime Authority (DMA). We provided the processed data, or you can download the raw data from DMA (https://dma.dk/safety-at-sea/navigational-information/ais-data)

**Skip this part** if you choose to use our processed data. For raw data preprocessing, please put the downloaded data into the data folder and kindly run the following scripts. (note the downloaded data file name should begin with "aisdk")
````
python is_preprocessing_noint.py 
````
````
python edit_preprocessing.py 
````
````
python traj2img.py
````

### Trajectory Similarity Measurement Module Training
````
python trajsim_train.py
````

### Route Recommendation Module Training
````
python recommendation_train.py
````

### Test
````
python recommendation_test.py  
````