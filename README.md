sentiment analyisis

This repo consist of  experiment to handle imbalanced datasets using various techniques and tracks the results with MLflow. The experiment processes text data from a CSV file, applies TF-IDF vectorization, and evaluates a RandomForestClassifier with different imbalance handling methods.

Features





Data Processing: Loads and preprocesses text data from csv file



Vectorization: Uses TfidfVectorizer with n-grams (1,3) and a maximum of 10,000 features.



Imbalance Handling: Implements five methods:





Class Weighting (class_weight='balanced')



SMOTE Oversampling



Random Undersampling



SMOTEENN (Combined SMOTE and Edited Nearest Neighbors)



ADASYN Oversampling



Model Training: Trains a RandomForestClassifier with 200 estimators and a max depth of 15.



Evaluation: Logs accuracy, classification report metrics (precision, recall, f1-score), and confusion matrices to MLflow.



Tracking: Uses MLflow with DagsHub for experiment tracking, logging parameters, metrics, and artifacts (confusion matrix plots).

Requirements





Python 3.12+



Libraries: mlflow, dagshub, imblearn, scikit-learn, pandas, numpy, seaborn, matplotlib



DagsHub account for MLflow tracking



Input data: processed_data.csv with clean_comment (text) and category (labels) columns

