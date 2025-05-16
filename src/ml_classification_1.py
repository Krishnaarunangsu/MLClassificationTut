import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
iris= load_iris()
X=iris.data # Features
y=iris.target # Target Labels
print(f'Features Names:{iris.feature_names}')
print(f'Target Classes:{iris.target_names}')

# Split Dataset into Train and Test
X_train, X_test, y_train, y_test=train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f'Training Samples:\n{X_train}')
print(f'Testing Samples:\n{X_test}')

# Train the model
model=LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Make predictions
y_pred=model.predict(X_test)

# Evaluate the model
accuracy_score=accuracy_score(y_test, y_pred)
classification_report=classification_report(y_test, y_pred)
confusion_matrix=confusion_matrix(y_test, y_pred)

# Display Results
print(f"Accuracy:{accuracy_score}")
print(f"Classification Report(for all classes):\n{classification_report}")
print(f"Confusion Matrix:\n{confusion_matrix}")


