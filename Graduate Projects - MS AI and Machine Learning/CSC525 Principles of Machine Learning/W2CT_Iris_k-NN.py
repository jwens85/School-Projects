#%%
#K Nearest Neighbor Classifier Project for Week 2 Critical Thinking Option 1 Iris Data
#John Wensink
#CSC525 Principles of Machine Learning
#Colorads State University - Global Campus
#Dr. Dong Nguyen
#May 25, 2025
#%%
#Cell 1 for data loading, inspection, feature/label separation
import pandas as pd

df = pd.read_csv("data/w2ct_iris.csv")

df["Name"].value_counts()
print(df["Name"].value_counts())

print("\nDataset Preview:")
print(df.head())

print("\nMissing values per column:")
print(df.isnull().sum())

features = df[["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]
species = df["Name"]

print("\nFeature sample:")
print(features.head())

print("\nLabel sample:")
print(species.head())

#(GeeksforGeeks, 2025)
#%%
#Cell 2 will import sklearn's StandardScalar preprocessing module and put our data into a Pandas DataFrame

#We want to normalize our z-score for this dataset so that our relatively large length measurements don't dominate our distance calculations compared to our relatively small width measurements
#We can standardize our data by rescaling absolute measurements as a function of average value and standard deviation instead, important when using k-NN which relies on Euclidian distances
#Since our dataset is purely numeric, with few outliers, and has a roughly Gaussian distribution StandardScaler is probably the best choice because we don't need outlier protection a la RobustScaler (Scikit-learn, n.d. -e)
from sklearn.preprocessing import StandardScaler as ss

#Declare the variable scaler as an instance of StandardScaler
scaler = ss()
#Declare the variable features_scaled as the standardized version of features using z-score normalization (value x - mean μ ) / standard deviation σ
features_scaled = scaler.fit_transform(features)

#Let's make a DataFrame to check our progress showing the first, middle, and last 3 samples to make sure our scaling was applied consistently across the targets
scaled_df = pd.DataFrame(features_scaled, columns=features.columns)

#Find the middle rows of our dataset and declare it as middle_rows
middle_index = len(scaled_df) // 2
middle_rows = scaled_df.iloc[middle_index -1 : middle_index + 2]

#Make a concatenated DataFrame showing the 3 first, middle, and last rows, and display it in our Jupyter notebook
pd.concat([
    pd.DataFrame(features_scaled, columns=features.columns).head(3),
    pd.DataFrame(middle_rows),
    pd.DataFrame(features_scaled, columns=features.columns).tail(3)
])
#%%
#Cell 3 we will handle our train/test split using sklearn's train_test_split (Scikit-learn, n.d. -f)
from sklearn.model_selection import train_test_split as tts

#A standard 80/20 split seems appropriate given that our data is already normalized, clean, and balanced across targets. No need to stratify or oversample/undersample like might be needed for real-world 'messy' data
features_train, features_test, species_train, species_test = tts(
    features_scaled, species, test_size=0.2
    #, random_state=13 #uncomment for reproducibility or leave commented for random train/test split
)

# Confirm that the training and testing sets have the correct shapes
print("Training features shape:", features_train.shape)
print("Testing features shape:", features_test.shape)
print("Training target shape:", species_train.shape)
print("Testing target shape:", species_test.shape)

# Show class distribution in training and test sets
print("\nClass distribution in training set:")
print(species_train.value_counts())

print("\nClass distribution in test set:")
print(species_test.value_counts())
#%%
#Cell 4 we will introduce the k-NN classifier from sklearn, we'll import the classifier, instantiate it with n_neighbors=5, and fit it on the training data

#Even though our features are numerical, our target is still categorical and as such a classifier is appropriate here instead of a regressor
from sklearn.neighbors import KNeighborsClassifier as knn #(Scikit-learn, n.d. -d)

#Instantiate and fit the k-NN model
model = knn(n_neighbors=5) #All other hyperparameters use default values (metric=minkowski, p=2, weights=uniform, algorithm='auto')
model.fit(features_train, species_train)

#Display the model's hyperparameters
print("k-NN Configuration:")
print(model.get_params())

#%%
#Cell 5 we will evaluate the trained k-NN model using the testing data. We'll use the test features to make predictions and compare them to the species labels

#Sklearn's accuracy_score is used to evaluate the model, it measures how often the model's predictions match the data's true label. Accuracy = correct predictions / total predictions. Since our data is balanced accuracy is appropriate, if it were imbalanced I might consider some combination of precision, recall, or F1 for a more useful measurement
from sklearn.metrics import accuracy_score #(Scikit-learn, n.d. -a)

#Just for fun let's also run a classification_report to measure precision, recall, F1, and support. I expect that it will be very close to accuracy due to our 'toy' dataset
from sklearn.metrics import classification_report #Scikit-learn, n.d.)

#Make predictions on the test features
kNN_guess = model.predict(features_test)

#Evaluate the model's prediction accuracy
accuracy = accuracy_score(species_test, kNN_guess)

#Print the results for accuracy
print(f"Model accuracy on test data: {accuracy: .3f}")

#Print the classification report results
print("\nClassification report:")
print(classification_report(species_test, kNN_guess))

#%%
#Cell 6 Import our textbooks visualization of the iris data from chapter 3 (Fenner, 2020, p 58)
import importlib
import IrisData_Visualization as plots
_ = importlib.reload(plots) #Reloads the library _ = prevents displaying the path

#These two pair plots from the iris dataset visualize how the features relate to one another and how well they separate the different iris species.

#The top plot shows how sepal and petal dimensions vary by species name, using color-coded classes. We can see that Setosa (blue) is clearly separable as it occupies its own distinct cluster for petal length and width. Versicolor (orange) and Virginica (green) overlap somewhat, with Virginica tendint to have longer and wider petals than Versicolor, without much difference in sepal length.

#The bottom plot is essentially the same data, but colored by target integer instead of species name. 0 = Setosa, 1 = Versicolor, and 2 = Virginica
#%%
# Cell 7: Visualize k-NN classification performance using a confusion matrix and bar plots of key metrics. I used an LLM (Grimoire, 2025) to help with some of the visualizations
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Generate the confusion matrix to compare predicted vs. actual labels
cm = confusion_matrix(species_test, kNN_guess, labels=model.classes_)

# Plot the confusion matrix as a heatmap (Scikit-learn, n.d. -c)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

#Print raw confusion matrix with labeled columns.
print("\nRaw Confusion Matrix:")
header = " " * 18 + "  ".join(f"{label:>15s}" for label in model.classes_)
print(header)
for actual_label, row in zip(model.classes_, cm):
    row_str = "  ".join(f"{val:>15d}" for val in row)
    print(f"{actual_label:>15s}  {row_str}")


#Extract precision, recall, and F1-score for each class from the classification report (Scikit-learn, n.d. -b)
report = classification_report(species_test, kNN_guess, output_dict=True)
metrics = ['precision', 'recall', 'f1-score']
classes = model.classes_

#Organize scores for each metric and class into a dictionary
scores = {metric: [report[label][metric] for label in classes] for metric in metrics}

#Prepare to plot grouped bar charts
x = range(len(classes))
width = 0.25

#Bar chart construction using MatPlotLib
plt.figure(figsize=(8, 5))
for i, metric in enumerate(metrics):
    plt.bar([p + width * i for p in x], scores[metric], width=width, label=metric)

# Configure axis labels and layout
plt.xticks([p + width for p in x], classes)
plt.ylim(0, 1.1)
plt.ylabel("Score")
plt.title("Precision, Recall, and F1-Score by Class")
plt.legend()
plt.tight_layout()
plt.show()

#Output the text scores used to make the plot
print("Scores used for plotting:\n")
for metric in metrics:
    print(f"{metric.capitalize()}:")
    for cls, score in zip(classes, scores[metric]):
        print(f"  {cls}: {score:.2f}")
    print()  # Blank line for readability


#Given the separation of the classes in these pair plots, we should expect the classifier to perform quite well. In both pair plots Setosa is completely isolated from the other two classes in feature space, and k-NN should be able to easily classify Setosa with near perfect accuracy, as its features do not overlap with those of Versicolor or Virginica. We might expect some confusion between Versicolor and Virginica, due to the overlap in sepal width vs sepal length. Expecting accuracy in the 90%s overall, precision/recall should be near 100% for Setosa and maybe a bit lower for Versiccolor and Virginica, and F1 will likely be highest for Setosa and a bot lower for the other two. Let's see what happens!

#%%
#References:
#Fenner, M. E. (2020). Machine learning with Python for everyone. Pearson Education.

#GeeksforGeeks. (2025, May 2). Exploratory Data Analysis on Iris Dataset.
#https://www.geeksforgeeks.org/exploratory-data-analysis-on-iris-dataset/

#Grimoire. (2025). OpenAI Grimoire GPT assistant [AI LLM].
#https://apps.apple.com/us/app/hivemind-grimoire/id6446332488

#Scikit-learn. (n.d. -a). Accuracy_score. Scikit-learn documentation.
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html

#Scikit-learn. (n.d. -b). ClassificationReport. Scikit-learn documentation.
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html

#Scikit-learn. (n.d. -c). ConfusionMatrixDisplay. Scikit-learn documentation.
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html

#Scikit-learn. (n.d. -d). KNeighborsClassifier. Scikit-learn documentation.
#https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

#Scikit-learn. (n.d. -e). StandardScaler. Scikit-learn documentation.
#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

#Scikit-learn. (n.d. -f). TrainTestSplit. Scikit-learn documentation.
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html