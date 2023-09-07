# Federated-Learning-Online-Main

## Introduction
This reposiroty contains datasets and files relevant to **Caching for Federated-Learning**. We have also made use of GitHub Repository of **DPP-CACHE** Algorithm for experimental comaprisions Links are given below.

- ***Kaggle***: https://www.kaggle.com/datasets/yoghurtpatil/311-service-requests-pitt
- ***DPP CACHE***: https://github.com/shashankp28/DPP-Cache

## How to run
To run our algorithm follow the below steps:

1. Install python dependencies.
```
pip install -r requirements.txt
```
2. Change environemt variables in the *.env* file. A sample is shown below.
```
Q_INIT = 0                                # Inital Q value.
PAST = 3                                  # Number of previous slots used to predict
V_0 = 500                                 # Coeffecient of O(sqrt(T))
FUTURE = 1                                # Number of future slots to predict
ALPHA = 0.1                               # Percentage of catalogue as cache
NUM_SEQ = 1000                             # Number of sequences
THRESHOLD = 423                           # Number of files in the catalogue
TRAIN_MEMORY = 5                          # Previous slots used to train
USE_SAVED = False                         # Whether to use saved model
COST_CONSTRAINT = 2                      # Fetching cost Constraint
TIME_LIMIT = inf                          # Maximum requests per slot
PATH_TO_INPUT = Datasets/311_dataset.txt  # Path to request dataset
```

***Note:*** **Keep FUTURE key to be always 1**

3. Run the following command
```
python fedAvgOnline.py
```
```
python dpp.py
```
## References
A Drift Plus Penalty Approach - https://github.com/shashankp28/DPP-Cache/blob/main/DPP_Cache_ICASSP2022.pdf
