import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('/content/titanic.csv')

# Display the dataset
print(data.describe(include='all'))

import missingno as msno

# Check for missing values
print(data.isnull().sum())
msno.matrix(data)
plt.show()

# Visualize distributions of categorical features
categorical_features = ['Sex', 'Pclass', 'Embarked']
for feature in categorical_features:
    sns.countplot(x=feature, data=data)
    plt.title(f'Distribution of {feature}')
    plt.show()

# Visualize survival rates by categorical features
for feature in categorical_features:
    sns.barplot(x=feature, y='Survived', data=data)
    plt.title(f'Survival Rate by {feature}')
    plt.show()

# Visualize survival counts
sns.countplot(x='Survived', data=data)

# Visualize survival counts by class
sns.countplot(x='Survived', hue='Pclass', data=data)

# Visualize class distribution by gender
sns.countplot(x='Pclass', hue='Sex', data=data)

# Visualize age distribution
sns.histplot(data['Age'], bins=40, color='orange')

# Visualize pairplot with hue by gender
sns.pairplot(data, hue='Sex', palette='dark')

# Fill missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data.drop('Cabin', axis=1, inplace=True)

# Convert categorical variables to dummy variables
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# Drop irrelevant columns
data.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

# Separate features and target variable
X = data.drop('Survived', axis=1)
y = data['Survived']

# Impute missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train K-Nearest Neighbors (KNN) model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
pred_knn = knn.predict(X_test)
print("Accuracy of Titanic Dataset using KNN =", accuracy_score(y_test, pred_knn))
print(classification_report(y_test, pred_knn))
print(confusion_matrix(y_test, pred_knn))

# Train Support Vector Classifier (SVC) model
from sklearn.svm import SVC
svc = SVC(kernel='rbf', probability=True)
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)
print("Accuracy of Titanic Dataset using SVC =", accuracy_score(y_test, pred_svc))
print(classification_report(y_test, pred_svc))

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Display sample predictions
sample_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
sample_data = X_test[sample_indices]
sample_actual = y_test.iloc[sample_indices]

sample_predictions = knn.predict(sample_data)
print("Sample Predictions with KNN:", sample_predictions)
print("Actual Labels:", sample_actual.values)

sample_predictions = svc.predict(sample_data)
print("Sample Predictions with SVC:", sample_predictions)
print("Actual Labels:", sample_actual.values)

sample_predictions = model.predict(sample_data)
print("Sample Predictions with Logistic Regression:", sample_predictions)
print("Actual Labels:", sample_actual.values)

# Create a DataFrame for sample data
original_columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
sample_data_df = pd.DataFrame(sample_data, columns=original_columns)
sample_data_df['Actual'] = sample_actual.values
sample_data_df['Predicted'] = sample_predictions
print("\nSample Data with Predictions and Actual Labels:\n", sample_data_df)

# Predict survival for new sample data
sample_data = [
    [3, 22.0, 1, 0, 7.25, 1, 0, 1],  # 3rd class, 22 years old, 1 sibling/spouse, 0 parent/child, fare 7.25, male, embarked S
    [1, 38.0, 1, 0, 71.2833, 0, 0, 1],  # 1st class, 38 years old, 1 sibling/spouse, 0 parent/child, fare 71.2833, female, embarked S
]

sample_data = scaler.transform(sample_data)  # Scale the sample data

sample_predictions = knn.predict(sample_data)
print("Sample Predictions with KNN:", sample_predictions)

sample_predictions = svc.predict(sample_data)
print("Sample Predictions with SVC:", sample_predictions)

sample_predictions = model.predict(sample_data)
print("Sample Predictions with Logistic Regression:", sample_predictions)

# ROC curve for KNN
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import cross_val_score

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

# ROC curve for Logistic Regression
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

# Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
plt.figure()
plt.plot(recall, precision, color='green', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Cross-validation scores
cv_scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Score:", cv_scores.mean())

# Confusion matrix heatmap
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()
