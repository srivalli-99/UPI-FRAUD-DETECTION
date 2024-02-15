# Install the Kaggle Python package
!pip install -q kaggle

# Import the 'files' module from the 'google.colab' library
from google.colab import files

# Prompt the user to upload a file (likely the Kaggle API token) from their local machine to Google Colab
files.upload()

# Create a directory named '.kaggle' in the home directory to store the Kaggle API token
!mkdir ~/.kaggle

# Copy the uploaded Kaggle API token file ('kaggle.json') to the '.kaggle' directory
!cp kaggle.json ~/.kaggle/

# Set the permissions of the Kaggle API token file ('kaggle.json') to read and write for the owner only
!chmod 600 ~/.kaggle/kaggle.json

# List all available datasets on Kaggle
!kaggle datasets list

# Download a specific dataset from Kaggle ('online-payment-fraud-detection') using its identifier
!kaggle datasets download -d jainilcoder/online-payment-fraud-detection

# Unzip the downloaded dataset file ('online-payment-fraud-detection.zip') in the current directory
!unzip online-payment-fraud-detection.zip


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv('/content/onlinefraud.csv')

# Data exploration
df.describe()
df.info()
df.nunique()
print("\nMissing Values:")
print(df.isnull().sum())

# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

print("Categorical Columns:", categorical_cols)
print("Numerical Columns:", numerical_cols)

# Visualizing Numeric Features - Histograms
for col in numerical_cols:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[col], bins=20, kde=True, color='skyblue')
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()

# Visualizing Categorical Columns - Bar plots
for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=col, data=df, palette='viridis')
    plt.title(f'Bar Plot of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.show()

# Correlation Matrix
correlation_matrix = df[numerical_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix for Online Payment Fraud Detection')
plt.show()

# Visualize the distribution of the target variable (fraud or non-fraud)
plt.figure(figsize=(8, 6))
sns.countplot(x='isFraud', data=df)
plt.title('Distribution of Fraud and Non-Fraud Transactions')
plt.xlabel('Fraud Label')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8, 8))
plt.pie(values, labels=labels, autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])
plt.title('Distribution of Fraud and Non-Fraud Transactions')
plt.show()

# Fraudulent vs. Non-Fraudulent Transactions
fraud_counts = df['isFraud'].value_counts(normalize=True) * 100
print("Percentage of Fraudulent Transactions:", fraud_counts[1])

# Transaction Types
plt.figure(figsize=(10, 6))
sns.countplot(x='type', data=df, palette='viridis')
plt.title('Distribution of Transaction Types')
plt.xlabel('Transaction Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
