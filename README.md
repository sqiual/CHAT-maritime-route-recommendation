# CHAT-maritime-trajectory-prediction
This project is the official implementation of CHAT: Maritime Route Prediction via Conditional Historical AIS Trajectory Data

The project is conducted based on data provided by Danish Maritime Authority (DMA) (https://dma.dk/safety-at-sea/navigational-information/ais-data) and the Office for Coastal Management (USA) (https://marinecadastre.gov/ais/)


### Data
Both the database mentioned in the paper and the training dataset for the CHAT model are provided through this link: https://sites.google.com/view/chat-database/home
For demo purpose, we only put the subset of our dataset under the ./data diectory. For reproducibility, please download the whole training dataset from the website provide above.

### Cutomize the area of interest
For demo purpose, the area of interest is selected to be Area 1 (DMA). To edit the area of interest, please edit the parameters in edit_hyper_parameters.json


### Data Preparation 
Prepare data for training, evaluation, and testing
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

