# Optimizing an ML Pipeline in Azure

## Overview
In this project, I constructed ML pipelines using scikit-learn, Hyperdrive, and AutoML. First, I constructed a pipeline with scikit-learn. Second, I configured a Hyperdrive run to find the optimal hyperparameters. Lastly, I configured an AutoML run to find an optimal model and set of hyperparameters. 

## Summary
I used the [UCI Bank Marketing dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing) to solve a classification problem. The classification goal is to predict if the client will subscribe to a term deposit with the bank. The best performing model was an XGBoost classifier instantiated by AutoML. 

## Scikit-learn Pipeline
The pipeline steps include cleaning data, balancing class, split the data into training/validation/test sets, training, testing and savings models. The classification algorithm is logistic regression, which is a binary classifier and one of the simplest models. The hyperparameter tuning focuses on two parameters: inverse of regularization strength and maximum number of iterations to converge.

I used RandomParameterSampling as the parameter sample. Parameter values are chosen from a set of discrete values or a distribution over a continuous range. The main benefit is the simplicity and relatively smaller number of runs. 

I used the BanditPolicy as the early stopping policy. Early termination is based on slack criteria, and a frequency and delay interval for evaluation. Any run that doesn't fall within the slack factor or slack amount of the evaluation metric with respect to the best performing run will be terminated. The benefit is that the current best run is used as the benchmark and you can customize how slack_factor and evaluation_interval. 

## AutoML
The model includes a MaxAbs scaler and a XGBoost classifier. The hyperparameters are the default values. 
![](screenshots/best_model.png)

## Pipeline comparison
The AutoML pipeline chose a more complex model that has a higher Weighted AUC score. The Hyperdrive pipeline used a very simple model that has a slightly lower AUC score.

## Future work
For the Hyperdrive pipeline, I will try a few models that are more complex. For the AutoML pipeline, I will experiment with more settings. 

## Proof of cluster clean up
![](delete/best_model.png)
