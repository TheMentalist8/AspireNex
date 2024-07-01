import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from prettytable import PrettyTable
from termcolor import colored

# Load the dataset
df = pd.read_csv('creditcard.csv')

# Normalize the 'Amount' and 'Time' columns
scaler = StandardScaler()
df[['Amount', 'Time']] = scaler.fit_transform(df[['Amount', 'Time']])

# Separate features and target variable
X = df.drop(columns=['Class'])
y = df['Class']

# Combine X and y into a single DataFrame for resampling
df_resampled = pd.concat([X, y], axis=1)

# Separate the majority and minority classes
df_majority = df_resampled[df_resampled.Class == 0]
df_minority = df_resampled[df_resampled.Class == 1]

# Undersample the majority class to balance the dataset
df_majority_undersampled = resample(
    df_majority, 
    replace=False,    # Sample without replacement
    n_samples=len(df_minority),    # To match minority class size
    random_state=42   # Ensure reproducibility
)

# Combine minority class with undersampled majority class
df_undersampled = pd.concat([df_majority_undersampled, df_minority])

# Separate features and target variable from the undersampled dataset
X_undersampled = df_undersampled.drop(columns=['Class'])
y_undersampled = df_undersampled['Class']

# Split the undersampled dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_undersampled, y_undersampled, test_size=0.3, random_state=42, stratify=y_undersampled
)

# Train Logistic Regression model
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Function to create a pretty table from the classification report
def create_pretty_table(report):
    # ANSI color codes
    RED = "\033[91m"
    GREEN = "\033[92m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    ENDC = "\033[0m"  # End color

    table = PrettyTable()
    table.field_names = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            table.add_row([
                f"{RED}{label}{ENDC}",
                f"{GREEN}{metrics['precision']:.2f}{ENDC}",
                f"{MAGENTA}{metrics['recall']:.2f}{ENDC}",
                f"{CYAN}{metrics['f1-score']:.2f}{ENDC}",
                f"{BLUE}{metrics['support']}{ENDC}"
            ])
    
    return table

# Generate classification reports
report_lr = classification_report(y_test, y_pred_lr, target_names=['Genuine', 'Fraudulent'], output_dict=True)
report_rf = classification_report(y_test, y_pred_rf, target_names=['Genuine', 'Fraudulent'], output_dict=True)

# Print the reports in a pretty table format
print("Logistic Regression Performance:")
print(create_pretty_table(report_lr))

print("Random Forest Performance:")
print(create_pretty_table(report_rf))

# Function to print precision, recall, and F1-score for both models with colors
def print_colored_metrics(model_name, precision, recall, f1_score):
    print(f"{model_name} Precision: {colored(f'{precision:.2f}', 'blue')}")
    print(f"{model_name} Recall: {colored(f'{recall:.2f}', 'green')}")
    print(f"{model_name} F1-score: {colored(f'{f1_score:.2f}', 'red')}")

# Print colored metrics for Logistic Regression
print_colored_metrics("Logistic Regression", precision_score(y_test, y_pred_lr), recall_score(y_test, y_pred_lr), f1_score(y_test, y_pred_lr))

# Print colored metrics for Random Forest
print_colored_metrics("Random Forest", precision_score(y_test, y_pred_rf), recall_score(y_test, y_pred_rf), f1_score(y_test, y_pred_rf))
