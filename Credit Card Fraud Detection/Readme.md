# Credit Card Fraud Detection

This project implements a machine learning solution for detecting fraudulent credit card transactions. It uses both Logistic Regression and Random Forest classifiers to identify potential fraud cases.


## Overview

Credit card fraud is a significant concern for financial institutions and consumers alike. This project aims to develop a model that can accurately identify fraudulent transactions, helping to prevent financial losses and protect customers.

## Requirements

- Python 3.7+
- pandas
- scikit-learn
- prettytable
- termcolor

## Installation

1. Clone this repository
2. Navigate to the project directory
3. Install the required packages

## Usage

1. Ensure your credit card transaction data is in a CSV file named 'creditcard.csv' in the project directory.
2. Run the script
3. The script will output the performance metrics for both Logistic Regression and Random Forest models.

## Methodology

1. Data Preprocessing: Normalize 'Amount' and 'Time' columns using StandardScaler.
2. Handle Class Imbalance: Undersample the majority class (non-fraudulent transactions) to balance the dataset.
3. Model Training: Train Logistic Regression and Random Forest classifiers on the balanced dataset.
4. Evaluation: Use precision, recall, and F1-score to evaluate model performance.

## Results

The script outputs a color-coded performance table for each model, showing:
- Precision
- Recall
- F1-Score
- Support

for both genuine and fraudulent transaction classifications.


## License

This project is open source and available under the MIT License
