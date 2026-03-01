# ============================================
# IRIS FLOWER CLASSIFICATION 🌸
# My First Kaggle Notebook
# ============================================

# --- 1. LIBRARIES ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# --- 2. LOAD DATA ---
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['flower'] = df['target'].map({0:'Setosa', 1:'Versicolor', 2:'Virginica'})

print("Shape:", df.shape)
print("\nFirst 5 rows:")
df.head()
# --- 3. EDA ---
print("Basic Info:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# Countplot
plt.figure(figsize=(6,4))
sns.countplot(x='flower', data=df, palette='Set2')
plt.title("Flower Distribution")
plt.show()

# Pairplot
sns.pairplot(df, hue='flower', palette='Set2')
plt.show()
# --- 4. MODEL TRAINING ---

# Features & Target
X = df[iris.feature_names]
y = df['target']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
# --- 5. EVALUATION ---

# Classification Report
print(classification_report(y_test, y_pred, 
      target_names=['Setosa','Versicolor','Virginica']))

# Confusion Matrix
plt.figure(figsize=(6,4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Setosa','Versicolor','Virginica'],
            yticklabels=['Setosa','Versicolor','Virginica'])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()
# --- 6. FEATURE IMPORTANCE ---
feat_imp = pd.Series(model.feature_importances_, index=iris.feature_names)
feat_imp.sort_values().plot(kind='barh', color='steelblue', figsize=(7,4))
plt.title("Feature Importance")
plt.show()
