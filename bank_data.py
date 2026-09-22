import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

bank = pd.read_csv("bank_data.csv")

bank.dropna(inplace=True)
 
features = ['person_income', 'loan_amnt', 'person_age', 'person_emp_length', 'loan_percent_income', 'person_home_ownership', 'loan_grade']

x = bank[features]
y = bank['loan_status']

x_encoded = pd.get_dummies(x)

train_x, val_x, train_y, val_y = train_test_split(x_encoded, y, test_size=0.2, random_state= 1)

scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)
val_x_scaled = scaler.transform(val_x)

models = []
models.append(('LogReg', LogisticRegression(max_iter=1000)))
models.append(('Tree',DecisionTreeClassifier(random_state= 1)))
models.append(('Forest', RandomForestClassifier(random_state= 1))) 
models.append(('KNN', KNeighborsClassifier()))
models.append(('NaiveBayes', GaussianNB()))
models.append(('SVM', SVC(gamma= 'auto')))

results = []
names = []


for name, model in models:
    Kfold = StratifiedKFold(n_splits=10, random_state= 1, shuffle= True)
    cv_scores = cross_val_score(model, train_x_scaled, train_y, cv= Kfold, scoring='accuracy')

    results.append(cv_scores)
    names.append(name)

    print(f"{name} Average Accuracy: {cv_scores.mean() * 100:.2f}%")


plt.boxplot(results, tick_labels= names)
plt.title('Algorithm Comparison')
plt.savefig('algorithm_comparison.png')

bank_model = RandomForestClassifier(random_state= 1)

bank_model = bank_model.fit(train_x_scaled, train_y)

val_prediction = bank_model.predict(val_x_scaled)

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
new_customer_scaled = scaler.transform(new_customer_encoded)
prediction = bank_model.predict(new_customer_scaled)

print("\n--- NEW CUSTOMER APPLICATION ---")
if prediction[0] == 1:
    print("AI Decision: LOAN DENIED (High Risk of Default)")
else:
    print("AI Decision: LOAN APPROVED (Safe)")