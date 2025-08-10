from __future__ import absolute_import, division, print_function, unicode_literals

#TF Warnings Suppressor import
import TF_Warnings_Suppressor
TF_Warnings_Suppressor.configure_tensorflow()

import os

import urllib.request
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#W4PM - Import EarlyStopping for Module 4 Requirement 1
from tensorflow.keras.callbacks import EarlyStopping

print("Using TensorFlow Version", tf.__version__)

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

#Step 1: Download the dataset using urllib.request as a workaround for keras.utils.get_file Content-Length bug
dataset_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
dataset_path = os.path.expanduser("data/auto-mpg.data")

#W4PM - Check if data already exists to avoid re-downloading
if not os.path.exists(dataset_path):
    print("Downloading dataset using urllib.request")
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    urllib.request.urlretrieve(dataset_url, dataset_path)
else:
    print("Dataset already exists, skipping download")

#Step 2: Import database using Pandas.
column_names = [
    'MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
    'Acceleration', 'Model Year', 'Origin'
]
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                          na_values="?", comment='\t',
                          sep=" ", skipinitialspace=True)
#(CSU-Global, n.d. -a)
#(GeeksforGeeks, 2025 -a)

#DataFrame best practice to make a copy of the raw dataset in case we were to manipulate the
#data, we could reference the raw dataset without the need to re-download
dataset = raw_dataset.copy()
#(Pandas, n.d.)

#Make a decision to drop any rows with missing data points. Simplest way to make sure all rows
#are complete, there are more sophisticated options available but .dropna is sufficient for this
#project use case (Wang, 2024)
#W4PM - Print dataset shape before and after cleaning
print(f"Dataset shape before dropping NaN: {dataset.shape}")
dataset = dataset.dropna()
#(W3Schools, n.d.)
print(f"Dataset shape after dropping NaN: {dataset.shape}")

#Step 4: Split the data into train and test.
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
#(CSU-Global, n.d. -a)

#Step 7: Review the statistics.
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
#(CSU-Global, n.d. -a)

#Step 10: Split features from labels.
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')
#(CSU-Global, n.d. -a)

#W4PM - Print dataset information
print(f"\nTraining samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Features: {list(train_dataset.columns)}")

#Step 11: Normalize the data using Z-Score Normalization
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
#(CSU-Global, n.d. -a)
#(GeeksforGeeks, 2025 -b)

#Step 12: Build the model.
#W4PM - Build improved model with dropout, L2 regularization, and Adam optimizer
def build_model():
    mpg_model = keras.Sequential([
        #W4PM - Increased neurons and added L2 regularization
        layers.Dense(128, activation='relu', 
                    input_shape=[len(train_dataset.keys())],
                    kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.Dropout(0.3),  #W4PM - Added dropout for regularization
        layers.Dense(64, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.Dropout(0.2),  #W4PM - Added dropout layer
        layers.Dense(32, activation='relu',  #W4PM - Added additional hidden layer
                    kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.Dense(1)
    ])
    
    #W4PM - Use Adam optimizer instead of RMSprop for better performance
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    mpg_model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])
    return mpg_model

model = build_model()
#(CSU-Global, 2025 -a)
#(CSU-Global, 2025 -b)

#Steps 13 & 14: Inspect the model, take a screenshot
#W4PM - Updated model architecture with improvements
print("\nImproved Model Architecture:")
model.summary()

#W4PM - Module 4 Requirement 1: Implement EarlyStopping callback
print("\nTraining with EarlyStopping")

#W4PM - Added restore_best_weights to keep the best model, min_delta to ignore tiny fluctuations
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True,
    min_delta=0.001
)

#(CSU-Global, n.d. -a)
#(CSU-Global, n.d. -b)
#(TensorFlow, n.d.)

#Steps 17 & 18: Train the model for 1000 epochs, and record the training and validation accuracy
#W4PM - Train with early stopping (will stop before 1000 if no improvement)
EPOCHS = 1000

#W4PM - Added early_stop callback to prevent overfitting
early_history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, 
    validation_split=0.2, 
    verbose=0,
    callbacks=[early_stop, tfdocs.modeling.EpochDots()]
)
#(CSU-Global, n.d. -b)
#(TensorFlow, n.d.)

#W4PM - Report when early stopping triggered
print(f"\n\nTraining stopped at epoch: {len(early_history.history['loss'])}")

#W4PM - Module 4 Requirement 2: Plot training history and analyze
plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
plotter.plot({'Early Stopping': early_history}, metric="mae")
plt.ylim([0, 10])
plt.ylabel('MAE [MPG]')
plt.title('Model Training with Early Stopping - MAE')
plt.savefig('early_stopping_mae_plot.png', dpi=150, bbox_inches='tight')
plt.show()
##(CSU-Global, n.d. -b)

#Get the final MAE from validation
final_val_mae = early_history.history['val_mae'][-1]
print(f"\nAverage Error (MAE) on validation set: {final_val_mae:.2f} MPG")
print("Quality Assessment: This error indicates the model's predictions are off by approximately "
      f"{final_val_mae:.2f} miles per gallon on average. For fuel efficiency prediction, "
      "this is a reasonable error margin for practical applications.")

#W4PM - Module 4 Requirement 3: Evaluate model on test set
print("\nMODULE 4 REQUIREMENT 3: Test Set Evaluation")
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
print(f"\nTesting set Mean Abs Error: {mae:5.2f} MPG")
#(CSU-Global, n.d. -b)

#W4PM - Module 4 Requirement 5: Make predictions and create scatter plot
print("\nMODULE 4 REQUIREMENT 5: Predictions and Scatter Plot")

test_predictions = model.predict(normed_test_data).flatten()
#(CSU-Global, n.d. -b)

#Create scatter plot
plt.figure(figsize=(8, 8))
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions, alpha=0.6)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims, 'r--', alpha=0.8)  #Perfect prediction line
#(CSU-Global, n.d. -b)

#Add R^2 score for quality assessment
from sklearn.metrics import r2_score
r2 = r2_score(test_labels, test_predictions)
plt.title(f'True vs Predicted MPG Values (R^2 = {r2:.3f})')
plt.savefig('prediction_scatter_plot.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nPrediction Quality Assessment:")
print(f"R^2 Score: {r2:.3f}")
print(f"Interpretation: The model explains {r2*100:.1f}% of the variance in MPG values.")
print("The scatter plot shows predictions clustered around the diagonal line, indicating good predictive performance.")

#W4PM - Module 4 Requirement 6: Analyze error distribution
print("\nMODULE 4 REQUIREMENT 6: Error Distribution Analysis")

error = test_predictions - test_labels
plt.figure(figsize=(10, 6))
plt.hist(error, bins=25, edgecolor='black', alpha=0.7)
plt.xlabel("Prediction Error [MPG]")
plt.ylabel("Count")
#(CSU-Global, n.d. -b)
plt.title("Distribution of Prediction Errors")


#Add statistics to the plot
mean_error = np.mean(error)
std_error = np.std(error)
plt.axvline(mean_error, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_error:.2f}')
plt.axvline(mean_error + std_error, color='orange', linestyle='dashed', linewidth=1, label=f'+/- 1 STD: {std_error:.2f}')
plt.axvline(mean_error - std_error, color='orange', linestyle='dashed', linewidth=1)
plt.legend()
plt.savefig('error_distribution_plot.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nError Distribution Analysis:")
print(f"Mean Error: {mean_error:.2f} MPG")
print(f"Standard Deviation: {std_error:.2f} MPG")
print(f"Error Range (95% confidence): [{mean_error - 2*std_error:.2f}, {mean_error + 2*std_error:.2f}] MPG")

#Check for normality
from scipy import stats
_, p_value = stats.normaltest(error)
print(f"\nNormality Test p-value: {p_value:.4f}")
if p_value > 0.05:
    print("The error distribution appears to be approximately normal (p > 0.05).")
    print("This suggests the model's errors are randomly distributed without systematic bias.")
else:
    print("The error distribution deviates from normal (p < 0.05).")
    print("This might indicate some systematic patterns in prediction errors.")

#W4PM - Additional analysis: Feature importance visualization
print("\nADDITIONAL ANALYSIS: Feature Relationships")

#Create a correlation matrix
correlation_matrix = train_dataset.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('feature_correlation_matrix.png', dpi=150)
plt.show()
#(DataCamp, n.d.)