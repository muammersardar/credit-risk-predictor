import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

bank = pd.read_csv("bank_data.csv")

bank.dropna(inplace=True)
 
features = ['person_income', 'loan_amnt', 'person_age', 'person_emp_length', 'loan_percent_income', 'person_home_ownership', 'loan_grade']

x = bank[features]
y = bank['loan_status']

x_encoded = pd.get_dummies(x)

train_x, val_x, train_y, val_y = train_test_split(x_encoded, y, test_size=0.2, random_state= 1)

bank_model = RandomForestClassifier(random_state= 1)

bank_model = bank_model.fit(train_x, train_y)

val_prediction = bank_model.predict(val_x)

accuracy = accuracy_score(val_y, val_prediction)

print(f'Model Accuracy: {accuracy * 100:.2f}%')

# --- THE LIVE PREDICTION ---

# 1. Create a brand new, fictional customer
new_customer = pd.DataFrame([{
    'person_income': 35000,
    'loan_amnt': 12000,
    'person_age': 24,
    'person_emp_length': 2,
    'loan_percent_income': 0.34,
    'person_home_ownership': 'RENT', # Renting
    'loan_grade': 'C'                # Average credit grade
}])

# 2. Translate the text into 1s and 0s just like before
new_customer_encoded = pd.get_dummies(new_customer)

# 3. Align the columns! (This ensures the new customer has the exact same binary switches as the training data)
new_customer_encoded = new_customer_encoded.reindex(columns=x_encoded.columns, fill_value=0)

# 4. Ask the AI to make a decision
prediction = bank_model.predict(new_customer_encoded)

print("\n--- NEW CUSTOMER APPLICATION ---")
if prediction[0] == 1:
    print("AI Decision: LOAN DENIED (High Risk of Default)")
else:
    print("AI Decision: LOAN APPROVED (Safe)")