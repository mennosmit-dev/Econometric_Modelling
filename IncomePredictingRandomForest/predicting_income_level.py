"""
Machine Learning Assignment for Income Level Prediction

This script performs the following steps:

1. Imports necessary libraries for data handling, visualization, and modeling.
2. Loads the dataset files including a codebook, training data, and test data.
3. Prepares the training dataset by separating features and target variable.
4. Uses nested cross-validation with Random Forest classifiers to:
   - Identify important features based on feature importance scores.
   - Tune hyperparameters including number of trees, max depth, and split criteria.
   - Evaluate out-of-sample accuracy for various feature subsets and hyperparameters.
   - Save the results of tuning to a CSV file.
5. Loads and aggregates tuning results to visualize the impact of hyperparameters
   and number of features on model accuracy.
6. Trains a final Random Forest model on the entire training dataset using
   the most important features identified.
7. Predicts the target variable on the test dataset using the final model.
8. Saves the predictions to a text file.

Notes:
- Standardizing or normalizing features is not required for Random Forest.
- Feature importance thresholding is used to select relevant features.
- Nested cross-validation ensures unbiased hyperparameter tuning.
"""

import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

# --- Load datasets ---

# Show all rows of dataframe
pd.set_option('display.max_rows', None)

# Load codebook for feature descriptions
codebook = pd.read_csv('codebook.csv')
print(codebook)

# Load training and test datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Separate features and target variable in training set
y_train = train['target']
X_train = train.drop('target', axis=1)

print(X_train.head(5))

# Note: Random Forests do not require feature scaling (standardizing/normalizing)

# --- Train model with nested cross-validation ---

# Dictionary to store out-of-sample accuracies (not used directly here but can be extended)
oos_accuracies = {}

# List to keep track of number of features used (optional)
num_features = []

# Setup outer k-fold cross-validation (5 splits)
outer_kf = KFold(n_splits=5, shuffle=True, random_state=0)

# Define hyperparameter grid for tuning
hyperparams = {
    'n_estimators': [100, 300],
    'max_depth': [10, None],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 5],
    'max_features': ['sqrt'],  # log2 tried but similar results
}

# Model criterion and feature importance threshold
criterion = 'entropy'
f_i_threshold = 0.010  # Features with importance below this are ignored

# Open CSV file to save cross-validation results
with open('oos_accuracies.csv', mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    # Write CSV header
    csv_writer.writerow([
        'fold',
        'n_estimators',
        'max_depth',
        'min_samples_split',
        'min_samples_leaf',
        'max_features',
        'num_features',
        'oos_accuracy',
    ])

    # Outer cross-validation loop with progress bar
    for fold_idx, (train_idx, test_idx) in enumerate(
        tqdm(outer_kf.split(X_train), desc="Outer CV", total=outer_kf.get_n_splits()), 1
    ):
        # Split data into current fold training and test sets
        X_train_fold = X_train.iloc[train_idx]
        y_train_fold = y_train.iloc[train_idx]
        X_test_fold = X_train.iloc[test_idx]
        y_test_fold = y_train.iloc[test_idx]

        # Fit a Random Forest to get feature importances
        rf = RandomForestClassifier(random_state=0)
        rf.fit(X_train_fold, y_train_fold)

        # Pair features with their importance scores
        feature_importances = list(zip(X_train_fold.columns, rf.feature_importances_))

        # Filter features by importance threshold
        important_features = [
            (feat, imp) for feat, imp in feature_importances if imp > f_i_threshold
        ]

        # Sort features by descending importance
        important_features.sort(key=lambda x: x[1], reverse=True)

        # Loop over increasing number of features starting from 25
        for num_feats in tqdm(range(25, len(important_features) + 1),
                              desc="Evaluating features",
                              leave=False):
            selected_features = [feat for feat, _ in important_features[:num_feats]]
            X_selected = X_train_fold[selected_features]

            # Grid search over hyperparameters
            for n_estimators in hyperparams['n_estimators']:
                for max_depth in hyperparams['max_depth']:
                    for min_samples_split in hyperparams['min_samples_split']:
                        for min_samples_leaf in hyperparams['min_samples_leaf']:
                            for max_features in hyperparams['max_features']:
                                # Initialize RF with current hyperparameters
                                rf = RandomForestClassifier(
                                    n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf,
                                    max_features=max_features,
                                    random_state=0,
                                    criterion=criterion,
                                )
                                # Evaluate accuracy with 5-fold CV on training fold
                                scores = cross_val_score(
                                    rf, X_selected, y_train_fold, cv=5, scoring='accuracy'
                                )
                                # Write results to CSV
                                csv_writer.writerow([
                                    fold_idx,
                                    n_estimators,
                                    max_depth,
                                    min_samples_split,
                                    min_samples_leaf,
                                    max_features,
                                    num_feats,
                                    np.mean(scores),
                                ])

# --- Visualize tuning results ---

# Load the saved results (assuming file is renamed to correct path)
results = pd.read_csv('oos_accuracies_result.csv')

# Drop the max_features column (not tuned finally)
results = results.drop(['max_features'], axis=1)

# Fill NaNs in max_depth for aggregation
results['max_depth'] = results['max_depth'].fillna("NaN")

# Define columns used for grouping
hyperparameters = ['num_features', 'n_estimators', 'max_depth',
                   'min_samples_split', 'min_samples_leaf']

# Aggregate results by hyperparameter combinations (mean accuracy)
results = results.groupby(hyperparameters).agg({'oos_accuracy': 'mean'}).reset_index()

# Create string representation of hyperparameter combinations for plotting
results['hyperparam_combo'] = results.apply(
    lambda row: (
        f"n_estimators={row['n_estimators']}, "
        f"max_depth={row['max_depth']}, "
        f"min_samples_split={row['min_samples_split']}, "
        f"min_samples_leaf={row['min_samples_leaf']}"
    ),
    axis=1
)

# Plot mean accuracy vs number of features for each hyperparameter combo
plt.figure(figsize=(10, 6))

unique_combos = results['hyperparam_combo'].unique()
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_combos)))

for i, combo in enumerate(unique_combos):
    subset = results[results['hyperparam_combo'] == combo]
    plt.plot(subset['num_features'], subset['oos_accuracy'],
             color=colors[i], label=combo, marker='o')

plt.xlabel('Number of Features')
plt.ylabel('Mean Accuracy')
plt.title('Mean Accuracy for Different Hyperparameter Combinations')
plt.legend(title='Hyperparameters', bbox_to_anchor=(0.5, -0.2),
           loc='upper center', fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig('mean_accuracy_hyperparameters.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Final model training and prediction ---

# Train Random Forest on full training data to get feature importances
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)

# Get and sort feature importances
feature_importances = list(zip(X_train.columns, rf.feature_importances_))
feature_importances.sort(key=lambda x: x[1], reverse=True)

# Select top 32 important features
top_features = feature_importances[:32]

# Plot feature importance bar chart
plt.barh([f[0] for f in top_features], [f[1] for f in top_features])
plt.xlabel('Feature Importance')
plt.title('Top 32 Important Features')
plt.gca().invert_yaxis()
plt.savefig('features_prediction.png', dpi=300, bbox_inches='tight')
plt.show()

# Prepare final training data with selected features
feature_names = [f[0] for f in top_features]
X_train_final = X_train[feature_names]

# Initialize final Random Forest model with chosen hyperparameters
rf_final = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=0,
    criterion='entropy'
)

# Fit final model on full training data
rf_final.fit(X_train_final, y_train)

# Prepare test set with same features
X_test_final = test[feature_names]

# Predict target for test data
y_pred = rf_final.predict(X_test_final)
print(y_pred)

# Save predictions to text file separated by spaces
with open('predictions.txt', 'w') as f:
    f.write(' '.join(map(str, y_pred)))
