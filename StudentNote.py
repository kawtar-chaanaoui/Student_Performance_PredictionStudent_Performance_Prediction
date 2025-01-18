
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


data = pd.read_csv("student-mat.csv", delimiter=";")

# Clean column names
data.columns = data.columns.str.strip().str.replace('"', '')

#------partie 1----------
# Print the first 5 rows
print("First 5 rows of the dataset:")
print(data.head())

# Descriptive statistics for G1, G2, and G3
print("\nDescriptive statistics for G1, G2, G3:")
print(data[['G1', 'G2', 'G3']].describe())

# Calculate mean and standard deviation for G3
mean_g3 = data['G3'].mean()
std_g3 = data['G3'].std()
print(f"\nMean of G3: {mean_g3:.2f}")
print(f"Standard deviation of G3: {std_g3:.2f}")

#------partie 2 ----------
# Histogram of G3
plt.figure(figsize=(8, 6))
sns.histplot(data['G3'], kde=True, bins=15, color='blue')
plt.title('Distribution of Final Grades (G3)')
plt.xlabel('G3 (Final Grade)')
plt.ylabel('Frequency')
plt.show()

# Scatter Plot: G1 vs G3
plt.figure(figsize=(8, 6))
sns.scatterplot(x='G1', y='G3', data=data, color='green')
plt.title('Relationship between G1 and G3')
plt.xlabel('G1 (First Trimester Grade)')
plt.ylabel('G3 (Final Grade)')
plt.show()

# Scatter Plot: G2 vs G3
plt.figure(figsize=(8, 6))
sns.scatterplot(x='G2', y='G3', data=data, color='orange')
plt.title('Relationship between G2 and G3')
plt.xlabel('G2 (Second Trimester Grade)')
plt.ylabel('G3 (Final Grade)')
plt.show()

#-----------
# --partie 3------------------
# Prepare data for modeling
X = data[['G1', 'G2']]
y = data['G3']

# Fit a linear regression model
model = LinearRegression()
model.fit(X, y)

# Evaluate the model
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
print(f"\nR² score of the model: {r2:.2f}")
print("The R² score indicates the proportion of variance in G3 explained by G1 and G2.")

# Make a prediction for G1=15 and G2=16
new_student = pd.DataFrame({'G1': [15], 'G2': [16]})
predicted_g3 = model.predict(new_student)
print(f"\nPredicted G3 for G1=15 and G2=16: {predicted_g3[0]:.2f}")

# Comment on prediction
print("The prediction appears reasonable, given the linear relationship observed.")

# Save notebook
print("\nEnsure this code is saved as 'NOM_Prenom_PredictionNotes.ipynb' for submission.")
