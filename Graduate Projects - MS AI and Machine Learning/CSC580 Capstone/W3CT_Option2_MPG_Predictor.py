from __future__ import absolute_import, division, print_function, unicode_literals

#TF Warnings Suppressor import
import TF_Warnings_Suppressor
TF_Warnings_Suppressor.configure_tensorflow()

import os
import urllib.request
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("Using TensorFlow Version", tf.__version__)

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
from tensorflow.keras.utils import plot_model

#Step 1: Download the dataset using urllib.request as a workaround for keras.utils.get_file Content-Length bug
dataset_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
dataset_path = os.path.expanduser("data/auto-mpg.data")
print("Downloading dataset using urllib.request")
urllib.request.urlretrieve(dataset_url, dataset_path)


#Step 2: Import database using Pandas.
column_names = [
    'MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
    'Acceleration', 'Model Year', 'Origin'
]
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                          na_values="?", comment='\t',
                          sep=" ", skipinitialspace=True)
#(CSU-Global, n.d.)
#(GeeksforGeeks, 2025 -a)

#DataFrame best practice to make a copy of the raw dataset in case we were to manipulate the
#data, we could reference the raw dataset without the need to re-download
dataset = raw_dataset.copy()
#(Pandas, n.d.)

#Make a decision to drop any rows with missing data points. Simplest way to make sure all rows
#are complete, there are more sophisticated options available but .dropna is sufficient for this
#project use case (Wang, 2024)
dataset = dataset.dropna()
#(W3Schools, n.d.)

#Step 3: Take a screenshot of the tail of the dataset.
print("Here is our dataset tail: \n", dataset.tail())

#Step 4: Split the data into train and test.
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
#(CSU-Global, n.d.)

#Step 5: Inspect the data.
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
#(CSU-Global, n.d.)
print("\nPlease refer to the Seaborn plot in your plots window for step 5\n")
#Output the Seaborn plot locally
plt.show()

#Step 6: Take a screenshot of the tail of the plots.
sns.pairplot(train_dataset.tail(5)[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
plt.show()

#Step 7: Review the statistics.
#Do not truncate output, give the DataFrame a bit of extra width to output all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
#(Marques, 2025)

train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
print(train_stats)
#(CSU-Global, n.d.)


#Step 8: Take a screenshot of the tail of the statistics.
print("\nTail of Training Data Statistics\n")
#Slice last 5 rows and create a copy
tail_stats = train_dataset[-5:].copy()
#(cottontail, 2013)

#Remove MPG column
tail_stats.pop("MPG")
#Get descriptive stats and transpose for correct orientation
tail_stats = tail_stats.describe().transpose()
print(tail_stats)

#Step 9: Prepare to split features from label column
label_column = 'MPG'
print(f"\nStep 9: Preparing to split features from label column: {label_column}\n")
print(train_dataset[[label_column]].head().reset_index(drop=True))

#Step 10: Split features from labels.
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')
#(CSU-Global, n.d.)

print("\nStep 10: Columns remaining in training dataset after label separation:\n")
print(train_dataset.columns.tolist())
#(Vultr, 2024)

#Step 11: Normalize the data using Z-Score Normalization
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
#(CSU-Global, n.d.)
#(GeeksforGeeks, 2025 -b)

print("\nStep 11: Preview of Normalized Training Data:\n")
print(normed_train_data.head(), "\n")

#Step 12: Build the model.
def build_model():
    mpg_model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    mpg_model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])
    return mpg_model

model = build_model()
#(CSU-Global, 2025)

#Steps 13 & 14: Inspect the model, take a screenshot
print("\nSteps 13 & 14: Inspecting the model architecture:\n")
model.summary()

#Graphviz and Keras plot_model()
print("\nGraphviz and Keras plot_model()\n")
plot_model(
    model,
    to_file='model_architecture.png',
    show_shapes=True,
    show_layer_names=True
)

#Step 15 & 16: Run the model with a batch size of 10 examples from the training and call model.predict
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print("\nInitial predictions on untrained model:\n", example_result)

#Steps 17 & 18: Train the model for 1000 epochs, and record the training and validation accuracy
EPOCHS = 1000

history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[tfdocs.modeling.EpochDots()]
)
#(CSU-Global, n.d.)

#Step 19 & 20: Visualize the model's training progress using the stats history stats
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
#(CSU-Global, n.d.)
print("\n",hist.tail())

plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
plotter.plot({'Basic': history}, metric="mae")
plt.ylim([0, 10])
plt.ylabel('MAE [MPG]')
#(CSU-Global, n.d.)
plt.show()

#Step 21: Provide a screenshot of the history plot.
plotter.plot({'Basic': history}, metric="mse")
plt.ylim([0, 20])
plt.ylabel('MSE [MPG^2]')
#(CSU-Global, n.d.)
plt.show()