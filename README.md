# Credit Risk Predictor

An end-to-end machine learning pipeline that predicts bank loan defaults using a Random Forest Classifier.

## Project Overview
This project evaluates customer financial data (such as income, loan amount, employment length, and home ownership status) to predict the likelihood of loan default. 

**Model Accuracy:** 91.13%

## Technologies Used
* **Python 3**
* **Pandas** (Data manipulation and One-Hot Encoding)
* **Scikit-Learn** (Random Forest classification and metrics)

## Dataset
The model was trained on historical banking data. 
*(Note: The raw dataset is excluded from this repository to maintain a lightweight environment. You can access the original data from [Kaggle's Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)).*

## How to Run
1. Clone this repository.
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the script: `python bank_data.py`