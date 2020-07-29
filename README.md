# Santander-Transaction-Prediction-with-GridSearchCV

## Data source: 
Santander Customer Transaction Prediction 
[Kaggle] https://www.kaggle.com/c/santander-customer-transaction-prediction/overview 

## Background:
Santander, a wholly-owned subsidiary of Spanish Santander Group, commits to helping people and businesses get better in finance. For these reasons, Kagglers was invited by Santander to help them find which clients would be able to make specific trades in the future.
In this challenge, the data provided by the contest has the same structure as the real data that solved the problem, but the data is anonymized, with each line containing 200 numeric values identified only by Numbers.

## Problem description:
Type: There are only two classes of the target, so this problem is a binary classification problem.
Features and target: All of them are not disclosure. Range of the target: {0, 1}, to represent {“not making a transaction”, “making a transaction”} respectively. All the features (counts 200) are continuous variable.
Sample: Majority is class 0 and minority is class 1, which means the sample is imbalanced.
Training and testing given: Size of 200k for both.

## Methods:
7 machine learning models have been applied into this project, including: Logistic Regression, Decision Tree, SVM, MLP, Gaussian Naive Bayes, Random Forest, XGBoost.  
Feature engineering: feature selection by Random Forest and dimension reduction by PCA.  
Then, applying sklearn Pipeline to wrap these models and running them with GridsearchCV to tune hyperparameters and obtain the best model.

