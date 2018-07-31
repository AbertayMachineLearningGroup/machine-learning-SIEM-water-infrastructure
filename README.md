# Improving SIEM for Critical SCADA Water Infrastructures Using Machine Learning

This work aims at using different machine learning techniques in detecting anomalies (including hardware failures, sabotage and cyber-attacks) in SCADA water infrastructure.

## Dataset Used
The dataset used is published [here](https://www.sciencedirect.com/science/article/pii/S2352340917303402) 

## Algorithms Used 
- Logistic Regression
- Gaussian Naive Bayes
- k-Nearest Neighbours
- Support Vector Machine
- Decision Trees
- Random Forests

## How to Run it:

```
Clone this repository
run preprocessing.py [dataset log path]
run classification.py [data processed file path]
run classification-with-confidence.py [data processed file path]
```
The output of preprocessing will be saved in the cloned directory as 'dataset_processed.csv'
The classification outputs is separated in folders for each output (anomaly, affected component, scenario, etc.). Each folder contains a csv for each algorithm having its confusion matrix and a 'CrossValidation.csv' file with the combined results.
```