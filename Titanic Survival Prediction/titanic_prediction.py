import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
data = pd.read_csv('/content/titanic.csv')

# Display basic statistics of the dataset
print(data.describe(include='all'))

import missingno as msno

# Check for missing values and visualize them
print(data.isnull().sum())
msno.matrix(data)
plt.show()

# Visualize distribution of categorical features
categorical_features = ['Sex', 'Pclass', 'Embarked']
for feature in categorical_features:
    sns.countplot(x=feature, data=data)
    plt.title(f'Distribution of {feature}')
    plt.show()

# Visualize survival rate by categorical features
for feature in categorical_features:
    sns.barplot(x=feature, y='Survived', data=data)
    plt.title(f'Survival Rate by {feature}')
    plt.show()

# Visualize survival count
sns.countplot(x='Survived', data=data)

# Visualize survival count by passenger class
sns.countplot(x='Survived', hue='Pclass', data=data)

# Visualize passenger class distribution by gender
sns.countplot(x='Pclass', hue='Sex', data=data)

# Visualize age distribution
sns.histplot(data['Age'], bins=40, color='orange')

# Create a pairplot to visualize relationships between features
sns.pairplot(data, hue='Sex', palette='dark')

# Handle missing values and drop unnecessary columns
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data.drop('Cabin', axis=1, inplace=True)

# Encode categorical variables
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# Drop irrelevant columns
data.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

# Separate features and target variable
X = data.drop('Survived', axis=1)
y = data['Survived']

# Impute missing values in features
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# K-Nearest Neighbors Classifier
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

pred_knn = knn.predict(X_test)
print("Accuracy of Titanic Dataset using KNN =", accuracy_score(y_test, pred_knn))
print(classification_report(y_test, pred_knn))
print(confusion_matrix(y_test, pred_knn))

# Support Vector Machine Classifier
from sklearn.svm import SVC

svc = SVC(kernel='rbf')
svc.fit(X_train, y_train)

pred_svc = svc.predict(X_test)
print("Accuracy of Titanic Dataset using SVC =", accuracy_score(y_test, pred_svc))
print(classification_report(y_test, pred_svc))

# Logistic Regression Classifier
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Make predictions on sample data
sample_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
sample_data = X_test[sample_indices]
sample_actual = y_test.iloc[sample_indices]

# KNN predictions
sample_predictions = knn.predict(sample_data)
print("KNN Sample Predictions:", sample_predictions)
print("Actual Labels:", sample_actual.values)

# SVC predictions
sample_predictions = svc.predict(sample_data)
print("SVC Sample Predictions:", sample_predictions)
print("Actual Labels:", sample_actual.values)

# Logistic Regression predictions
sample_predictions = model.predict(sample_data)
print("Logistic Regression Sample Predictions:", sample_predictions)
print("Actual Labels:", sample_actual.values)

# Display sample data with predictions
original_columns = [
    'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S'
]
sample_data_df = pd.DataFrame(sample_data, columns=original_columns)
sample_data_df['Actual'] = sample_actual.values
sample_data_df['Predicted'] = sample_predictions
print("\nSample Data with Predictions and Actual Labels:\n", sample_data_df)

# Make predictions on new data
sample_data = [
    [3, 22.0, 1, 0, 7.25, 1, 0, 1],  # 3rd class, 22 years old, male, embarked S
    [1, 38.0, 1, 0, 71.2833, 0, 0, 1],  # 1st class, 38 years old, female, embarked S
]

sample_data = scaler.transform(sample_data)  # Scale the sample data

# KNN predictions on new data
sample_predictions = knn.predict(sample_data)
print("KNN Sample Predictions:", sample_predictions)

# SVC predictions on new data
sample_predictions = svc.predict(sample_data)
print("SVC Sample Predictions:", sample_predictions)

# Logistic Regression predictions on new data
sample_predictions = model.predict(sample_data)
print("Logistic Regression Sample Predictions:", sample_predictions)

# ROC Curve for KNN
y_pred_prob = knn.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='green', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) for KNN')
plt.legend(loc="lower right")
plt.show()

# ROC Curve for Logistic Regression
y_pred_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='pink', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='yellow', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) for Logistic Regression')
plt.legend(loc="lower right")
plt.show()

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)

plt.figure()
plt.plot(recall, precision, color='green', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Score:", cv_scores.mean())

# Confusion Matrix Heatmap
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()
