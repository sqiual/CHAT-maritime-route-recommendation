# CHAT-maritime-trajectory-prediction
This project is the official implementation of CHAT: Maritime Route Prediction via Conditional Historical AIS Trajectory Data

The project is conducted based on data provided by Danish Maritime Authority (DMA) (https://dma.dk/safety-at-sea/navigational-information/ais-data) and the Office for Coastal Management (USA) (https://marinecadastre.gov/ais/)


### Data
Both the database mentioned in the paper and the training data for the CHAT model are provided through this link: https://sites.google.com/view/chat-database/home

Please download the files from "CHAT - Training Data," unzip them, and place them under the ./data/ directory.
Also unzip the trained model parameters file under the ./checkpoint/ directory.


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

