import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

df = pd.read_csv("data_core.csv")
df = df.drop(columns=["Fertilizer Name", "Soil Type"], axis=1)

print("Dataset Preview:")
print(df.head())
print("\nDataset Information:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())
print("\nCrop Distribution:")
print(df['Crop Type'].value_counts())

X = df.drop('Crop Type', axis=1)
y = df['Crop Type']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Store the mapping for later reference
crop_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
print("\nCrop Type Mapping:")
for crop, code in crop_mapping.items():
    print(f"{crop}: {code}")

# Simple data augmentation by adding small random variations
augmented_data = []
for _ in range(5):  # Create 5 variations for each sample
    for i in range(len(df)):
        row = df.iloc[i].copy()
        # Add small random variations to numerical features
        for col in ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']:
            # Add variation of up to ±5% of the original value
            variation = np.random.uniform(-0.05, 0.05) * row[col]
            row[col] += variation
        augmented_data.append(row)

# Create augmented dataframe
augmented_df = pd.DataFrame(augmented_data)
# Combine with original data
combined_df = pd.concat([df, augmented_df], ignore_index=True)

# Check the size of the augmented dataset
print(f"\nAugmented dataset size: {len(df)}")
print("Augmented crop distribution:")
print(combined_df['Crop Type'].value_counts())

# Prepare the features and target from the augmented data
X = combined_df.drop('Crop Type', axis=1)
y = combined_df['Crop Type']

# Encode the labels for the augmented dataset
y_encoded = label_encoder.transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded)


# Define a function to evaluate a classifier
def evaluate_classifier(name, classifier, X_train, X_test, y_train, y_test, scaler=None):
    if scaler:
        X_train_processed = scaler.fit_transform(X_train)
        X_test_processed = scaler.transform(X_test)
    else:
        X_train_processed = X_train
        X_test_processed = X_test


    classifier.fit(X_train_processed, y_train)

    y_pred = classifier.predict(X_test_processed)

    accuracy = accuracy_score(y_test, y_pred)

    if scaler:
        pipeline = Pipeline([
            ('scaler', scaler),
            ('classifier', classifier)
        ])
        cv_scores = cross_val_score(pipeline, X, y_encoded, cv=5)
    else:
        cv_scores = cross_val_score(classifier, X, y_encoded, cv=5)

    return {
        'Name': name,
        'Classifier': classifier,
        'Accuracy': accuracy,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std(),
        'Predictions': y_pred
    }


scaler = StandardScaler()

classifiers = [
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42)),
    ('XGBoost', XGBClassifier(random_state=42)),
    ('Support Vector Machine', SVC(random_state=42)),
    ('K-Nearest Neighbors', KNeighborsClassifier()),
    ('Neural Network', MLPClassifier(max_iter=1000, random_state=42)),
    ('Decision Tree', DecisionTreeClassifier(random_state=42)),
    ('Extra Trees', ExtraTreesClassifier(random_state=42)),
    ('AdaBoost', AdaBoostClassifier(random_state=42)),
    ('Naive Bayes', GaussianNB()),
    ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42))
]

results = []
for name, clf in classifiers:
    print(f"\nEvaluating {name}...")
    result = evaluate_classifier(name, clf, X_train, X_test, y_train, y_test, scaler=scaler)
    results.append(result)
    print(f"Test Accuracy: {result['Accuracy']:.4f}")
    print(f"Cross-validation Score: {result['CV Mean']:.4f} (±{result['CV Std']:.4f})")

results_df = pd.DataFrame([{
    'Classifier': r['Name'],
    'Test Accuracy': r['Accuracy'],
    'CV Mean Accuracy': r['CV Mean'],
    'CV Standard Deviation': r['CV Std']
} for r in results])

results_df = results_df.sort_values('CV Mean Accuracy', ascending=False).reset_index(drop=True)

print("\nClassifier Performance Comparison:")
print(results_df)

plt.figure(figsize=(12, 8))
sns.barplot(x='Test Accuracy', y='Classifier', data=results_df, color='skyblue')
plt.title('Classifier Performance Comparison (Test Accuracy)')
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig('classifier_comparison_accuracy.png')

plt.figure(figsize=(12, 8))
plt.errorbar(
    x=results_df['CV Mean Accuracy'],
    y=results_df['Classifier'],
    xerr=results_df['CV Standard Deviation'],
    fmt='o',
    capsize=5
)
plt.axvline(x=results_df['CV Mean Accuracy'].max(), color='red', linestyle='--')
plt.title('Classifier Performance Comparison (Cross-Validation)')
plt.xlabel('Mean Accuracy (with std dev)')
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig('classifier_comparison_cv.png')

best_idx = results_df['CV Mean Accuracy'].idxmax()
best_classifier_name = results_df.loc[best_idx, 'Classifier']
best_classifier = next(clf for name, clf in classifiers if name == best_classifier_name)
print(f"\nBest classifier: {best_classifier_name}")

best_result = next(r for r in results if r['Name'] == best_classifier_name)

print(f"\nDetailed report for {best_classifier_name}:")
print("\nClassification Report:")

y_test_crops = label_encoder.inverse_transform(y_test)
y_pred_crops = label_encoder.inverse_transform(best_result['Predictions'])
print(classification_report(y_test_crops, y_pred_crops))

cm = confusion_matrix(y_test_crops, y_pred_crops)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title(f'Confusion Matrix for {best_classifier_name}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('best_classifier_confusion_matrix.png')


def tune_hyperparameters(classifier_name, classifier, X, y, scaler):
    print(f"\nPerforming hyperparameter tuning for {classifier_name}...")
    
    param_grid = {}
    
    # Define parameter grids based on classifier type
    if classifier_name == 'Random Forest':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif classifier_name == 'Gradient Boosting':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7, 10]
        }
    elif classifier_name == 'XGBoost':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7, 10],
            'subsample': [0.8, 1.0]
        }
    elif classifier_name == 'Support Vector Machine':
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
    elif classifier_name == 'K-Nearest Neighbors':
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }
    
    
    if not param_grid:
        print(f"No parameter grid defined for {classifier_name}. Skipping tuning.")
        return classifier
    
    # Create pipeline with scaler and classifier
    pipeline = Pipeline([
        ('scaler', scaler),
        ('classifier', classifier)
    ])
    
    # Create GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        {f'classifier__{key}': val for key, val in param_grid.items()},
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the grid search
    grid_search.fit(X, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Return the best estimator
    return grid_search.best_estimator_.named_steps['classifier']

if len(combined_df) >= 50:  # Only tune if we have a reasonable amount of data
    best_tuned_classifier = tune_hyperparameters(best_classifier_name, best_classifier, X, y, scaler)
    
    # Compare tuned vs untuned model
    untuned_result = best_result
    
    # Evaluate tuned model
    tuned_result = evaluate_classifier(
        f"Tuned {best_classifier_name}",
        best_tuned_classifier,
        X_train, X_test, y_train, y_test,
        scaler=scaler
    )
    
    print("\nComparison of original vs tuned model:")
    print(f"Original {best_classifier_name} CV Score: {untuned_result['CV Mean']:.4f}")
    print(f"Tuned {best_classifier_name} CV Score: {tuned_result['CV Mean']:.4f}")
    
    # Save the best model
    import pickle
    with open('best_crop_recommendation_model.pkl', 'wb') as f:
        pickle.dump(best_tuned_classifier, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
else:
    # Save the best model without tuning
    import pickle
    with open('best_crop_recommendation_model.pkl', 'wb') as f:
        pickle.dump(best_classifier, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

print("\nModel evaluation and selection completed!")

# Function to predict crop based on the best model
def predict_crop(temperature, humidity, moisture, nitrogen, potassium, phosphorous, model, scaler):
    # Create a DataFrame with the input parameters
    input_data = pd.DataFrame({
        'Temparature': [temperature],
        'Humidity': [humidity],
        'Moisture': [moisture],
        'Nitrogen': [nitrogen],
        'Potassium': [potassium],
        'Phosphorous': [phosphorous]
    })
    
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    
    # If the model has predict_proba method
    if hasattr(model, 'predict_proba'):
        # Get prediction probabilities
        proba = model.predict_proba(input_scaled)
        
        # Get all crops with their probabilities
        crop_probs = list(zip(model.classes_, proba[0]))
        crop_probs.sort(key=lambda x: x[1], reverse=True)
        
        return prediction[0], crop_probs
    else:
        return prediction[0], None

# Load the best model (this would typically be done in a separate file)
with open('best_crop_recommendation_model.pkl', 'rb') as f:
    best_model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Example usage
print("\nExample prediction using the best model:")
crop, probabilities = predict_crop(
    temperature=28.0,
    humidity=52.0,
    moisture=43.0,
    nitrogen=33,
    potassium=4,
    phosphorous=4,
    model=best_model,
    scaler=scaler
)

print(f"Predicted crop: {crop}")
if probabilities:
    print("Crop probabilities:")
    for crop_name, prob in probabilities:
        print(f"  {crop_name}: {prob:.4f}")
