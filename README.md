Reddit Sentiment Analysis Project

This repository contains Jupyter notebooks for a sentiment analysis project on Reddit comments. The project involves data preprocessing, handling imbalanced datasets, and tracking experiments using MLflow with DagsHub integration. The goal is to classify comments into categories (e.g., positive, negative, neutral) while addressing class imbalance issues.

Notebooks





pre.ipynb: Handles data loading, cleaning, null value removal, duplicate handling, and basic exploratory data analysis (EDA) including word clouds for different sentiment categories.



imbalanced_experiment.ipynb (inferred from the MLflow code): Implements experiments for handling imbalanced data using techniques like class weighting, oversampling (SMOTE, ADASYN), undersampling, and combined methods (SMOTEENN). Uses TF-IDF vectorization, RandomForestClassifier, and MLflow for logging parameters, metrics, and artifacts.

Features





Data Preprocessing (pre.ipynb):





Loads data from Reddit_Data.csv with columns clean_comment (text) and category (sentiment labels: 1 for positive, 0 for neutral, -1 for negative).



Handles missing values (drops rows with NaN in clean_comment).



Removes duplicates.



Generates word clouds for visualization of common words in positive, negative, and neutral comments.



Imbalanced Data Handling & Experiments (imbalanced_experiment.ipynb):





TF-IDF vectorization with n-grams (1-3) and max 10,000 features.



Stratified train-test split (80-20).



Imbalance techniques: Class weighting, SMOTE, Random Undersampling, SMOTEENN, ADASYN.



Model: RandomForestClassifier (200 estimators, max depth 15).



Evaluation: Accuracy, precision, recall, F1-score per class; confusion matrices saved as artifacts.



MLflow tracking: Logs parameters (e.g., ngram_range, sampler), metrics, and confusion matrix plots to DagsHub.

Requirements





Python 3.12+



Libraries: pandas, numpy, nltk, matplotlib, seaborn, mlflow, dagshub, imbalanced-learn, scikit-learn



