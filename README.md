# Citibike-Analysis
## Code Documentations
```
preprocess/
   --
   -- 
   --
Model/
  - baseline.ipynb: code for baseline tree
  - RNN/ 
     -- dataset.py: Citibike dataset
     -- data_organize.py: organize data from 1 year csv into month/day/12 csvs
     -- model.py: GCN, RNN, CitiBike_Model
     -- train.py: used to train the model (using MSE), created in model.py. Notice, that each time the cuurent version of code is saved.
     -- eval.py : used to eval the mdoel (using MSE).


```
## Project Overview and Methods

The goal of the Citibike analysis is to use the citibike data publised at https://s3.amazonaws.com/tripdata/index.html for the year of 2021, and build prediction models to predict the cluster-imbalanceness for every 2 hours.

The procedure includes the follows:

1. Preprocess the data: delete data such as those 
                        ```contains any nulls; 
                        
                        ride duration > 24 hours (i.e. start and end on dfferent dates);
                        
                        for each day, the record does not cover the entire 24 hours;
                        
                        ect..
                        ```
2. Cluster the stations using kmeans clustering
3. Define station "Inbalancenesss" as the inbound - outbound per 2 hour window.
4. Build transition matrix every 2 hours. Within each matrix, entry A[i,j] denotes the number of bikes depart from i to j within this 2 hour chunk.
5. Cluter the stations to reduce matrix sparcity.
6. Build two models(baseline and RNN) to $\mapsto$ $6\times6$ matrix to $1\times6$ vector for every 2 hours.


## Flowchart

![overflow.png](Overflow.png)
