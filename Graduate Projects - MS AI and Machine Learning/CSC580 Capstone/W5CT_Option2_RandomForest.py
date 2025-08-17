#~~~Step 1: Load the data~~~

#Import required libraries
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
#(CSU-Global, n.d.)

#Import optional libraries
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

np.random.seed(13)

#Declare the variable iris, and load the built-in iris dataset from sklearn
iris = load_iris()
#(CSU-Global, n.d.)

#Create a DataFrame named df containing the feature data and column lables from the iris dataset
df = pd.DataFrame(iris.data, columns=iris.feature_names)
#(CSU-Global, n.d.)

#Prevent truncation/wrapping of wide DataFrames
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#Display the first 5 rows of the DF using df.head() to preview the loaded feature data
print("Initial dataframe (top 5 rows):")
print(df.head())
print()
#(CSU-Global, n.d.)

#Add a new column with the species names; this is what we are going to try to predict
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
#(CSU-Global, n.d.)

#View the top 5 rows
print("Dataframe with species column (top 5 rows):")
print(df.head())
#(CSU-Global, n.d.)
print("\nSCREENSHOT 1: Take a screenshot of the head of the dataset\n")

#~~~Step 2: Create training and test data~~~

#Add a new column called is_train that assigns each row a random value between 0 and 1
#If the value <= 0.75, mark the row as a training row by setting a Boolean value as True
#ELSE the row gets a False tag and will be used as a testing row. This is our train/test split
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
#(CSU-Global, 2025)
#(Nelamali, 2024)

#View the top 5 rows
print("Dataframe with is_train column (top 5 rows):")
print(df.head())
print()

#Create 2 separate DFs for training and testing by filtering rows where is_train is True or False
train, test = (
    df[df['is_train'] == True],
    df[df['is_train'] == False]
)
#(CSU-Global, n.d.)

#Confirm our train/test split is working as intended by printing the lengths of the train/test observations
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:', len(test))
#(CSU-Global, n.d.)
print("\nSCREENSHOT 2: Take a screenshot of the outputs\n")

#~~~Step 3: Preprocess the data~~~

#Pull the first four column names of our dataset to isolate the independent variables we will be using
#We can use Python's slice notation to easily accomplish this
features = df.columns[:4]
#(CSU-Global, n.d.)

#Output our selected feature columns to ensure we're looking at the correct columns
print("Features:")
print(features.tolist())
print()

#Convert species strings into numerical categories
y = pd.factorize(train['species'])[0]
#(CSU-Global, n.d.)

#Calculate the middle 10 values to be able to print the first, middle, and last n (10) values
midpoint = len(y) // 2
start = midpoint - 5
end = midpoint + 5

#View target values
print("Target (y) - first 10 values:")
print("Our target value has", len(y), "values")
print("Here are the first 10 target values", y[:10])
print("Here are the middle 10 target values:", y[start:end])
print("Here are the last 10 target values", y[-10:])
print("\nSCREENSHOT 3: Take a screenshot of the outputs\n")

#~~~Step 4: Train the random forest classifier~~~

#Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_jobs=2, random_state=0)
#(CSU-Global, n.d.)

#Train the classifier using the feature data and their corresponding feature labels
clf.fit(train[features], y)
#(CSU-Global, n.d.)
#(DataCamp, 2024)

print("Random Forest Classifier trained successfully!\n")

#Print out the model parameters in chunks of 4 parameters per line
params = clf.get_params()
param_items = list(params.items())
print("Model parameters:")
for i in range(0, len(param_items), 4):
    chunk = param_items[i:i+4]
    print("  " + ", ".join(f"{k}={v}" for k, v in chunk))
#(Grimoire, 2025)

print()

#~~~Step 5: Apply the classifier to the test data~~~

#Run the classifier model on our test data
predictions = clf.predict(test[features])
#(CSU-Global, n.d.)

#Get predicted probabilities for the first 10 observations
predicted_probs = clf.predict_proba(test[features])[:10]

print("Test Dataset Species Counts:\n",test['species'].value_counts())
print("\nPredicted probabilities for the first 10 test observations:\n")
print("(Columns represent probability for each species: setosa, versicolor, virginica)")
for i, probs in enumerate(predicted_probs):
    print(f"Observation {i+1}: {probs}")
print("\nSCREENSHOT 4: Take a screenshot of the predicted probabilities\n")

#~~~Step 6: Evaluate the classifier by comparing predicted and actual species~~~

#Get actual species for test data
actual_species = test['species'].values[:5]
predicted_species = clf.predict(test[features])[:5]

#Map predictions back to species names
species_names = iris.target_names
predicted_species_names = [species_names[pred] for pred in predicted_species]

print("Comparison of actual vs predicted species for first 5 test observations:")
for i in range(5):
    print(f"Observation {i+1}: Actual = {actual_species[i]}, Predicted = {predicted_species_names[i]}")
print()

#~~~Step 7: Create a confusion matrix~~~

#Get all predictions for test data
all_predictions = clf.predict(test[features])
preds = pd.Categorical.from_codes(all_predictions, iris.target_names)

#Create confusion matrix
conf_matrix_df = pd.crosstab(test['species'], preds,
                             rownames=['Actual Species'],
                             colnames=['Predicted Species'])
#(CSU-Global, n.d.)

print("Confusion Matrix:")
print(conf_matrix_df)
print("\nSCREENSHOT 5: Take a screenshot of the confusion matrix\n")

#Optional Seaborn Confusion Matrix
cm = confusion_matrix(test['species'].cat.codes, all_predictions)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()
#(Marsja, 2023)

#Calculate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test['species'].cat.codes, all_predictions)
print(f"Model Accuracy: {accuracy:.2%}")
print()

#~~~Step 8: View the list of features and their importance scores~~~

#Examine which features contribute most to prediction accuracy
feature_importance = list(zip(train[features].columns, clf.feature_importances_))
#(CSU-Global, n.d.)

print("Feature Importance Scores:")
for feature, importance in feature_importance:
    print(f"{feature}: {importance:.4f}")
print()

#Sort features by importance
sorted_features = sorted(feature_importance, key=lambda x: x[1], reverse=True)
print("Features ranked by importance:")
for i, (feature, importance) in enumerate(sorted_features, 1):
    print(f"{i}. {feature}: {importance:.4f}")
print("\nSCREENSHOT 6: Take a screenshot of feature importance\n")

#Additional model evaluation metrics
from sklearn.metrics import classification_report

print("Detailed Classification Report:")
print(classification_report(test['species'], preds, 
                           target_names=['setosa', 'versicolor', 'virginica']))
print("Random Forest Classifier Analysis Complete!")