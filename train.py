import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle


data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

df = X.copy()
df['target'] = y
df.to_csv('data.csv', index=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)


model = SVC(kernel='rbf', C=1, gamma='scale', probability=True)
model.fit(X_train_scaled, y_train)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)



with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)