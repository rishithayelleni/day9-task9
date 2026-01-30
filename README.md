
Detect fraudulent credit card transactions using machine learning techniques with a focus on handling imbalanced data.


Source: Kaggle Credit Card Fraud Dataset  
Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  

The dataset contains transactions made by European cardholders.
- Class 0: Normal transaction
- Class 1: Fraud transaction

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

1. Loaded the dataset and analyzed class imbalance.
2. Separated features and target variable.
3. Standardized the Amount column.
4. Used stratified train-test split.
5. Trained a Logistic Regression model as baseline.
6. Trained a Random Forest classifier.
7. Evaluated models using precision, recall, and F1-score.
8. Visualized feature importance.
9. Saved the trained Random Forest model.

Fraud datasets are highly imbalanced. Accuracy may appear high even if fraud cases are not detected properly. Precision and recall give better insight.

- random_forest_fraud_model.pkl (saved model)
- Feature importance plot


1. Download dataset from Kaggle.
2. Place creditcard.csv in project folder.
3. Install required libraries.
4. Run:
   python fraud_detection.py

- Understanding ensemble learning
- Handling imbalanced datasets
- Model evaluation beyond accuracy
