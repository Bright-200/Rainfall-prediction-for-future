import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load data
df = pd.read_csv("weatherAUS.csv")

# Drop rows with missing target
df = df.dropna(subset=['RainTomorrow'])

# Binary encode target
df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})

# Select relevant columns
features = ['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustDir',
            'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
            'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
            'Temp9am', 'Temp3pm', 'RainToday']

target = 'RainTomorrow'

X = df[features]
y = df[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify column types
categorical_cols = ['Location', 'WindGustDir', 'RainToday']
numerical_cols = list(set(features) - set(categorical_cols))

# Pipelines
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# Final pipeline with model
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=5, random_state=42))
])

# Train
model_pipeline.fit(X_train, y_train)

# Save model and metadata
joblib.dump({
    'model': model_pipeline,
    'numerical_cols': numerical_cols,
    'categorical_cols': categorical_cols
}, 'Rain.joblib')

print("✅ Model trained and saved as 'Rain.joblib'")
