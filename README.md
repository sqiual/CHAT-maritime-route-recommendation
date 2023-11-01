# CHAT-maritime-trajectory-prediction
This project is the official implementation of CHAT: Maritime Route Prediction via Conditional Historical AIS Trajectory Data

The project is conducted based on data provided by Danish Maritime Authority (DMA) (https://dma.dk/safety-at-sea/navigational-information/ais-data) and the Office for Coastal Management (USA) (https://marinecadastre.gov/ais/)

**Feel free to explore our full paper (CHAT_full_version.pdf) for a more comprehensive examination of the details and analytical results.**

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

<img width="1262" alt="Screenshot 2023-11-02 at 04 25 27" src="https://github.com/sqiual/CHAT-maritime-route-recommendation/assets/96676550/5f2e1903-23b7-4f77-b614-a194c8ffda29">


