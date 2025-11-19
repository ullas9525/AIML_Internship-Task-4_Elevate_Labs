import pandas as pd # Import pandas for data manipulation
import numpy as np # Import numpy for numerical operations
import os # Import os for operating system interactions (e.g., creating directories)
import matplotlib.pyplot as plt # Import matplotlib for plotting
import seaborn as sns # Import seaborn for enhanced visualizations

from sklearn.model_selection import train_test_split # Import train_test_split for splitting data
from sklearn.preprocessing import StandardScaler # Import StandardScaler for feature scaling
from sklearn.linear_model import LogisticRegression # Import LogisticRegression for the model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score # Import metrics for model evaluation

# Define paths for data and output to make the script flexible
DATASET_PATH = 'Dataset/data.csv' # Path to the dataset CSV file
OUTPUT_DIR = 'Output' # Directory to save output plots

# Create output directory if it doesn't exist to store plots
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load the breast_cancer.csv dataset
try:
    df = pd.read_csv(DATASET_PATH) # Load the dataset into a pandas DataFrame
    print("Dataset loaded successfully.") # Confirm successful dataset loading
except FileNotFoundError:
    print(f"Error: Dataset not found at {DATASET_PATH}. Please ensure the file exists.") # Handle file not found error
    exit() # Exit the script if the dataset is not found

# 2. Basic preprocessing
# Drop any ID / unnamed columns not useful for prediction
df = df.drop(columns=[col for col in df.columns if 'ID' in col or 'Unnamed' in col], errors='ignore') # Remove columns containing 'ID' or 'Unnamed'

# Handle missing values if they exist by dropping rows with any missing values
df.dropna(inplace=True) # Remove rows with any NaN values from the DataFrame
print(f"Shape after dropping missing values: {df.shape}") # Show DataFrame shape after handling missing values

# Separate features (X) and target (y) for binary classification
# Assuming 'diagnosis' is the target column based on common breast cancer datasets
X = df.drop('diagnosis', axis=1) # Features are all columns except 'diagnosis'
y = df['diagnosis'].map({'M': 1, 'B': 0}) # Target 'diagnosis': Malignant (M) -> 1, Benign (B) -> 0

# Check class distribution to understand the balance of target variable
print("Class distribution:\n", y.value_counts()) # Display the count of each class in the target variable

# 3. Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Split data with 80% train, 20% test, stratified

# 4. Standardize the features
scaler = StandardScaler() # Initialize StandardScaler for feature scaling
X_train_scaled = scaler.fit_transform(X_train) # Fit scaler on training data and transform it
X_test_scaled = scaler.transform(X_test) # Transform test data using the fitted scaler

# 5. Train a LogisticRegression model
model = LogisticRegression(random_state=42, solver='liblinear') # Initialize Logistic Regression model with a random state
model.fit(X_train_scaled, y_train) # Train the model on the scaled training data

# 6. Use the model to predict
y_pred = model.predict(X_test_scaled) # Predict class labels for the test set
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] # Predict probabilities of the positive class for the test set

# 7. Evaluate the model
conf_matrix = confusion_matrix(y_test, y_pred) # Calculate the confusion matrix
accuracy = accuracy_score(y_test, y_pred) # Calculate the accuracy score
precision = precision_score(y_test, y_pred) # Calculate the precision score
recall = recall_score(y_test, y_pred) # Calculate the recall score
f1 = f1_score(y_test, y_pred) # Calculate the F1-score
roc_auc = roc_auc_score(y_test, y_pred_proba) # Calculate the ROC-AUC score

print("\n--- Model Evaluation ---") # Section header for model evaluation
print(f"Confusion Matrix:\n{conf_matrix}") # Print the calculated confusion matrix
print(f"Accuracy: {accuracy:.4f}") # Print the accuracy score, formatted
print(f"Precision: {precision:.4f}") # Print the precision score, formatted
print(f"Recall: {recall:.4f}") # Print the recall score, formatted
print(f"F1-score: {f1:.4f}") # Print the F1-score, formatted
print(f"ROC-AUC Score: {roc_auc:.4f}") # Print the ROC-AUC score, formatted

# 8. Generate and SAVE the following plots
# Confusion matrix heatmap
plt.figure(figsize=(8, 6)) # Create a new figure with a specified size
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, # Create a heatmap of the confusion matrix
            xticklabels=['Predicted 0', 'Predicted 1'], # Set x-axis tick labels
            yticklabels=['Actual 0', 'Actual 1']) # Set y-axis tick labels
plt.title('Confusion Matrix') # Set the title of the plot
plt.ylabel('Actual Label') # Set the y-axis label
plt.xlabel('Predicted Label') # Set the x-axis label
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_heatmap.png')) # Save the plot to the output directory
plt.close() # Close the plot to free up memory

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba) # Calculate False Positive Rate, True Positive Rate, and thresholds
plt.figure(figsize=(8, 6)) # Create a new figure with a specified size
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})') # Plot the ROC curve with AUC score
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Plot the random classifier line
plt.xlim([0.0, 1.0]) # Set the x-axis limits
plt.ylim([0.0, 1.05]) # Set the y-axis limits
plt.xlabel('False Positive Rate') # Set the x-axis label
plt.ylabel('True Positive Rate') # Set the y-axis label
plt.title('Receiver Operating Characteristic (ROC) Curve') # Set the title of the plot
plt.legend(loc="lower right") # Display the legend at the lower right corner
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve.png')) # Save the plot to the output directory
plt.close() # Close the plot to free up memory

# Histogram/curve showing the sigmoid / probability distribution
plt.figure(figsize=(8, 6)) # Create a new figure with a specified size
sns.histplot(y_pred_proba[y_test == 0], color='blue', label='Class 0 (Benign)', kde=True, stat='density', alpha=0.5) # Plot probabilities for actual Class 0
sns.histplot(y_pred_proba[y_test == 1], color='red', label='Class 1 (Malignant)', kde=True, stat='density', alpha=0.5) # Plot probabilities for actual Class 1
plt.title('Predicted Probability Distribution by Actual Class') # Set the title of the plot
plt.xlabel('Predicted Probability of Malignant') # Set the x-axis label
plt.ylabel('Density') # Set the y-axis label
plt.legend() # Display the legend
plt.savefig(os.path.join(OUTPUT_DIR, 'probability_distribution.png')) # Save the plot to the output directory
plt.close() # Close the plot to free up memory

# 9. Demonstrate threshold tuning
print("\n--- Threshold Tuning ---") # Section header for threshold tuning

thresholds_to_try = [0.3, 0.5] # Define a list of thresholds to evaluate

for threshold in thresholds_to_try: # Iterate through each threshold
    y_pred_threshold = (y_pred_proba >= threshold).astype(int) # Convert probabilities to class labels based on the current threshold
    precision_t = precision_score(y_test, y_pred_threshold) # Calculate precision for the current threshold
    recall_t = recall_score(y_test, y_pred_threshold) # Calculate recall for the current threshold
    print(f"\nThreshold: {threshold}") # Print the current threshold being evaluated
    print(f"  Precision: {precision_t:.4f}") # Print the precision for the current threshold
    print(f"  Recall: {recall_t:.4f}") # Print the recall for the current threshold

# Print a short text summary explaining how changing threshold affects precision vs recall
print("\nSummary of Threshold Tuning:") # Summary header for threshold tuning
print("  - A lower threshold (e.g., 0.3) increases recall (identifying more positive cases) at the cost of precision (more false positives).") # Explain effect of lower threshold
print("  - A higher threshold (e.g., 0.5) increases precision (fewer false positives) but might decrease recall (missing some positive cases).") # Explain effect of higher threshold
print("  - The optimal threshold depends on the specific problem's requirements, balancing false positives and false negatives.") # Conclude on optimal threshold
