from sklearn.datasets import load_wine
import pandas as pd

data = load_wine()
X = pd.DataFrame(data['data'], columns = data['feature_names'])
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
