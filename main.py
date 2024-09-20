# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.tree import plot_tree
import zipfile
import os

# Step 1: Extract the .zip file and load the dataset
zip_file_path = r'c:\Users\harin\Documents\Custom Office Templates\bank-additional.zip'

# Extract the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall('bank_data')  # Extracts to a folder named 'bank_data'

# Load the CSV file from the extracted folder
csv_file_path = os.path.join('bank_data', 'bank-additional', 'bank-additional-full.csv')
data = pd.read_csv(csv_file_path, sep=';')

# Step 2: Perform basic data exploration
print(data.head())
print(data.info())
print(data.describe())

# Step 3: Handle categorical data using Label Encoding
label_encoder = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = label_encoder.fit_transform(data[column])

# Step 4: Define the features (X) and target variable (y)
X = data.drop(columns=['y'])  # All columns except the target
y = data['y']  # Target column

# Step 5: Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Initialize the Decision Tree Classifier
classifier = DecisionTreeClassifier(random_state=42)

# Step 7: Train the model on the training data
classifier.fit(X_train, y_train)

# Step 8: Make predictions on the test data
y_pred = classifier.predict(X_test)

# Step 9: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 10: Visualize the Decision Tree
plt.figure(figsize=(20,10))
plot_tree(classifier, filled=True, feature_names=X.columns, class_names=['no', 'yes'], rounded=True)
plt.show()

