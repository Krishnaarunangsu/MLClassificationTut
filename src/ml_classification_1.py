import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris= load_iris()
X=iris.data # Features
y=iris.target # Target Labels
print(f'Features Names:{iris.feature_names}')
print(f'Target Classes:{iris.target_names}')

