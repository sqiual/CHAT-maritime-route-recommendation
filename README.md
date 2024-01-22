# CHAT-maritime-trajectory-prediction
This project is the official implementation of CHAT: Maritime Route Prediction via Conditional Historical AIS Trajectory Data

The project is conducted based on data provided by Danish Maritime Authority (DMA) (https://dma.dk/safety-at-sea/navigational-information/ais-data) and the Office for Coastal Management (USA) (https://marinecadastre.gov/ais/)


### Data
Both the database mentioned in the paper and the training dataset for the CHAT model are provided through this link: https://sites.google.com/view/chat-database/home

For demo purpose, we only provide a subset of our dataset under ./data. For reproducibility, please download the whole training and testing dataset from the website provided above.



### Data Preparation 
Prepare data for training, evaluation, and testing. Make sure you have the 
- "dma_traj_array.hdf5" (for training and eval)
- "dma_traj_array_db.hdf5" (for training and eval)
- "dma_traj_array_test.hdf5" (for testing)
- "dma_traj_array_test_db.hdf5" (for testing)
  
files download from the website provided above, and correctly placed under ./data
````
python divide_dataset.py
python divide_dataset_test.py 
````

### Training
````
python ceshi_train.py
````

### Testing
````
python ceshi_test.py
````

