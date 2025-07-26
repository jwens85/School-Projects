print("Welcome to the Video Game Future Revenue Predictor!")
print()

import TF_Warnings_Suppressor

#Simple script to reduce clutter from the output related to TF-related warnings and
#logs from TF's C++ backend, oneDNN optimizations, gRPC, GLOG, and Autograph. The function
#configure_tensorflow() function sets TF's logger to error-only mode, and prints what device
#TF is using for execution, hopefully the GPU
TF_Warnings_Suppressor.configure_tensorflow()
print()

#We will need to import OS to check whether csv files exist using os.path.exists()
import os
#We will need to import Pandas to manipulate the csv data using DataFrames
import pandas as pd
#We will need to import MinMaxScaler to normalize numerical data so that all features fall
#proportionally within the 0-1 range
from sklearn.preprocessing import MinMaxScaler
#It might be nice to investigate our dataset's r2 score to measure how well a linear model might
#perform inference on unseen data
from sklearn.metrics import r2_score
#We will need to import LinearRegression to build a baseline model that predicts total earnings
#using a simple linear relationship between features
from sklearn.linear_model import LinearRegression
#We will need to import MSE as explicitly required by the assignment. It's a pretty good regression
#metric for our dataset, because it has a high r2 score. It would maybe be easier to interpret
#a mean absolute error, but the assignment requires we use MSE which is sufficient for this ddata
from sklearn.metrics import mean_squared_error
#While not required, simple linear regression might not be the best regression method to evaluate
#performance, a decision tree can capture some nonlinear relationships that a simple LR model might
#miss
from sklearn.tree import DecisionTreeRegressor
#Throwing a RandomForrestRegressor into the mix might be useful to compare how an ensemble
#method performs against a single decision tree, and the LR model. If RandomForest performs well, it
#might suggest that our dataset has more complex nonlinear patterns that benefit from aggregation
#across multiple learners
from sklearn.ensemble import RandomForestRegressor
#We'll use NumPy to sort the random forest's feature importance analysis section so we can rank the
#features from most to least important in a clean and efficient manner
import numpy as np
#We will need to import Sequential to define the architecture of our NN step by step, and
#load_model to reload the trained model later for making predictions on unseen data without
#the need to retrain for every inference
from tensorflow.keras.models import Sequential, load_model
#We are required to use Dense layers for this project. This is pretty standard for fully
#connected layers in a feedforward NN. Each neuron in a Dense layer receives input from all
#neurons in the previous layer, which is suitable for regression tasks like this one. Other options
#like Conv1D/2D, recurrent layers like LSTM or GRU are inappropriate and overkill for our dataset
from tensorflow.keras.layers import Dense
#Since we're going to be comparing regressors, it would be nice to have some visualizations to
#communicate the results more intuitively than just looking at the raw numbers alone. It's harder
#to get a sense of proportion when you're just looking at small decimal loss results, and visualizing
#performance using just basic histograms would go a long way toward understanding which models
#and features are contributing toward our predictive accuracy
import matplotlib.pyplot as plt

#~~~#
#PART 1: Load and Scale Data

#Load training data set from the downloaded csv file. It will get saved in our W2CT data subfolder under
#the project root's data folder. We might consider using something like os.path.join() to allow for
#some portability across different operating systems' path separators if this were going into a production
#environment, but for the purposes of a school project, hard-coded paths are fine
training_data_df = pd.read_csv("data/W2CT_Sales/sales_data_training.csv")
test_data_df = pd.read_csv("data/W2CT_Sales/sales_data_test.csv")

#Here's where we satisfy the project requirement to normalize the dataset by scaling values to the 0-1
#range using MinMaxScaler. I experimented with other scaling functions like RobustScaler and MaxAbsScaler
#but the differences in MSE were negligible, The project requirements seem to require MinMaxScaler, and
#it performs well enough that we'll just keep to the example
scaler = MinMaxScaler(feature_range=(0, 1))

#Scale both the training inputs and outputs. What's happening here is that we're using fit_transform(...)
#where fit calculates the min/max of each feature, transform then scales each value into the 0-1 range
#using those stats, and the result gets stored in the variable we declare here scaled_testing/training
scaled_training = scaler.fit_transform(training_data_df)
scaled_testing = scaler.transform(test_data_df)

#Print out the adjustment that the scaler applied to the total_earnings column of data
print("Total Earnings values were scaled by multiplying by {:.10f} and adding {:.6f}".format(scaler.scale_[8],
scaler.min_[8]))
print()
#The LLM (Grimoire, 2025) helped me to clean up my print statement here to make it cleaner and more readable


#Quick r2 check on our training data to confirm that MSE is an appropriate error metric by testing
#how well a simple linear model fits the scaled features. R2 tells us how much of the variation in total
#earnings can be explained by the input features. A high r2 means that the data fits a linear model well,
#and helps to confirm that using MSE as our evaluation metric makes sense. If r2 were low, it might suggest
#that the relationship is nonlinear or simply too weak for MSE to be meaningful. For this dataset, we see
#r2 of .9408, which suggests a strong, but maybe incomplete linear relationship
X = training_data_df.drop('total_earnings', axis=1).values
Y = training_data_df['total_earnings'].values
baseline_model = LinearRegression()
baseline_model.fit(X, Y)
predictions = baseline_model.predict(X)
r2 = r2_score(Y, predictions)
print("Baseline Linear Regression r2 on scaled training set: {:.4f}".format(r2))

#Create new pandas DataFrames from the scaled data to preserve the original column names and structure.
#This will make our data easier to work with later, especially when saving to csv or selecting specific
#columns like total_earnings to focus on. Using a DataFrame keeps the data organized and human-readable
#compared to a raw NumPy array
scaled_training_df = pd.DataFrame(scaled_training, columns=training_data_df.columns.values)
scaled_testing_df = pd.DataFrame(scaled_testing, columns=test_data_df.columns.values)

#Now we're glad that we made a DataFrame object, because it's super easy to just call the function
#to_csv(path, index=false) to save our cleaned and scaled data directly to a csv without having to
#do any additional formatting or conversion
scaled_training_df.to_csv("data/W2CT_Sales/scaled_train.csv", index=False)
scaled_testing_df.to_csv("data/W2CT_Sales/scaled_test.csv", index=False)

#~~~#
#PART 2: Keras Neural Network

#Load the scaled training data
training_data_df = pd.read_csv("data/W2CT_Sales/scaled_train.csv")
#We're going to want to isolate the input features from the target variable here. The model needs to learn
#a mapping from inputs (X) to outputs (Y), since we've put our training data into a handy DataFrame, it's
#quite easy to simply use df.drop(label, axis).values to ensure the model learns from the data's other
#features like genre, pricing, exclusivity, etc.
X = training_data_df.drop('total_earnings', axis=1).values
Y = training_data_df[['total_earnings']].values

#We'll be using Keras' Sequential() model for this assignment which is appropriate when we want to stack
#layers one after another in a straight line, from input to output. It's also required by the assignment
model = Sequential()

#Here's where we'll define the NN architecture, conforming specifically to the project's required
#specifications of using a sequential model with 9 input features, dense layers, ReLU activation
#for the hidden layers, and a single output layer with linear activation for regression. Keras makes it easy.
#The arguments required to add a layer are units (number of neurons in the layer), input_dim for the first layer
#(the number of input features), and the activation function to apply to the layer's output, in this case,
#the assignment requires ReLU for the hidden layers, and linear for the output layer. The assignment's
#requirements are pretty clear on what activation functions to use, but this is not the only way to skin this cat.
#ReLU is a strong default because it's computationally efficient and helps a little to avoid vanishing gradients,
#but alternatives like Leaky ReLU or ELU can improve training stability on real world datasets where we're using
#deeper networks or have particularly noisy data. Similarly, while a linear output is pretty standard for a
#regression problem like this, other functions like sigmoid or softmax might be used when the target output
#is a probability or a classification. Given the project's requirements, the architecture selected is valid,
#but they represent just one configuration among many valid architectural choices depending on the data's
#characteristics and the project's goals.
model.add(Dense(units=64, input_dim=9, activation='relu'))  #First hidden layer
model.add(Dense(units=32, activation='relu'))  #Second hidden layer
model.add(Dense(units=1, activation='linear'))  #Output layer with linear activation

#Here we're going to compile our model. It's important to disambiguate what we mean by 'compile' in this context.
#We know Python is an interpreted language, and we're not 'compiling' from source code into machine code here, rather,
#we're referring to configuring our Keras model for training. This includes selecting the optimizer which will
#determine how weights are updated. In this case, we're using the Adam optimizer because it's a reliable optimizer
#that works well across a wide variety of NN architectures. Adam's pretty good, because it combines the advantages
#of AdaGrad and RMSProp by adapting the learning rate for each parameter individually and using momentum to smooth
#updates. Models will tend to converge faster this way and be more stable during training, especially on noisy
#gradients or sparse data, which makes it appropriate for small or medium size regression tasks like this one.
model.compile(optimizer='adam', loss='mean_squared_error')

#Train the Keras Sequential NN by using model.fit() to construct and execute a directed acyclic graph (DAG), where
#each node represents a TF operation. Seeing model.fit() here can appear to be deceptively simple, there's a lot
#going on under the hood when we call this function including forward passes, loss computation, gradient computation,
#weight updates via backpropagation, all being managed within TF's DAG framework. This line is doing a lot of heavy
#lifting that isn't obvious by the simplicity and elegance of the function. We are compelled to follow the assignment
#requirements here training for 50 epochs, shuffling the data at random for each epoch, and dialing the verbosity
#to 2 to give per-epoch training summaries without completely overwhelming the output console with batch-level output
print("\nTraining Keras Sequential Neural Network Model ")
history = model.fit(X, Y, epochs=50, shuffle=True, verbose=2)

#Same idea as loading the training dataset into a DataFrame, here we'll do the same with the test dataset
test_data_df = pd.read_csv("data/W2CT_Sales/scaled_test.csv")

#Here we prepare our test set by splitting it into features (X_test) and target (Y_test). Just like with the training
#data, we drop our target (total_earnings) to isolate the dependent variable from the columns used to make predictions
X_test = test_data_df.drop('total_earnings', axis=1).values
Y_test = test_data_df[['total_earnings']].values

#Here we're evaluating the trained model on the test set by using model.evaluate(), which runs the forward pass
#on X_test, compares the predictions to Y_test, and returns the MSE loss. We'll declare test_error_rate here
#to store the result, and replace verbosity with a custom print line since we really only care about the MSE for
#this project. This keeps the console output clean while complying with project requirements to test performance.
test_error_rate = model.evaluate(X_test, Y_test, verbose=0)
print()
print("***MSE for the test data set is***: {}".format(test_error_rate))
print()

#Save the model to disk as per project requirements using the 'legacy' HDF5 format.
model.save("data/W2CT_Sales/trained_model.h5")
print("HDF5 Model saved to disk.")

#We've suppressed TF warnings for this file, but if we hadn't TF would gripe that our .hd5 model is a legacy format
#and that while still widely supported, the modern way to save models is in .keras format. So, to that end, we will
#'get with the times' here and also save the model using the modern Keras v3 format, which is probably better. This
#is because the modern format includes additional metadata for things like model architecture, training history,
#preprocessing layers, external dependencies, versioning details, and any model assets explicitly added using the
#model.add_metric() or model.add_loss() functions. Although we're not using any model assets in this project, it's
#probably best practice to save the model in Keras format for forward compatability, and h5 format for backward
#compatability.
model.save("data/W2CT_Sales/trained_model.keras", save_format="keras")
print("Keras Model saved to disk.")
print()

#~~~#
#PART 3: Make Predictions About a Proposed New Product

#Load the trained HDF5 model
model = load_model('data/W2CT_Sales/trained_model.h5')

#Load the synthetically created row of data for our 'proposed_new_product' that we wish to speculate for future revenue
proposed_file = "data/W2CT_Sales/proposed_new_product.csv"

#This block checks for the existence of a csv file containing synthetic feature data for our proposed game. If the
#file exists, it's loaded into a DF and the 'game_id' column is dropped, since it's a non-feature identifier not used
#by our model. This way only the relevant numeric input features are passed forward
if os.path.exists(proposed_file):
    X = pd.read_csv(proposed_file).drop('game_id', axis=1).values
    print("Loaded proposed_new_product.csv")
    print()
else:
    #If the file doesn't exist (you have to create it yourself, it's not in the LinkedIn dataset),
    print("The file for the proposed_new_product.csv was not found. Using hardcoded test input.")
    print()
    X = [[3.9, 0, 1, 0, 1, 0, 1, 0, 49.99]]

#Here, model.predict(X) takes the input array X (which contains our proposed product's features) and returns a NumPy
#array of predictions. Even though there's only one input sample, the output will still be a 2D array shape (1,1)
#because it's going to treat the prediction like a batched output.
prediction = model.predict(X)

#Since we just want the scalar prediction value from this nested structure, we'll need to access the first element of
#the first (only) row using prediction[0][0]
prediction = prediction[0][0]

#Now, we'll need to rescale the data from the 0-1 range back to Dollars. These constants are from when the data was
#originally scaled down using our MinMax scaler. This block is essentially just reversing the preprocessing that
#normalized our target output by multiplying the prediction by the scale factor used for the 9th column (index 8)
#and then adding back the original minimum value. This restores the prediction back to its original scale in Dollars
#and will allow us to format it in a human-readable output with just a bit of formatting on the print line below
prediction = prediction - scaler.min_[8]
prediction = prediction / scaler.scale_[8]

#So, to print the prediction in Dollars format ($xxx,xxx,xxx.xx) we can format our output string using {:, .2f}
#This tells Python to round the number to two decimal places (.2f) and insert commas as thousands separators (,).
#It's actually quite nice that Python automatically can group digits in sets of three, starting from the decimal and
#moving left to follow the USA style of number formatting. Of course, this is locale dependant, if we were outputting
#to Indian Rupees, it'd behoove us to import a library like format_currency from something like babel.numbers to give
#us their regional standard. Example ₹1,00,00,000, or ten million Rupees, AKA one Crore. Just an interesting tidbit.
print()
print("***Earnings Prediction for Proposed Product*** - ${:,.2f}".format(prediction))

#~~~#
#Part 4: Compare Performance for the Different Types of Regressors and the NN
print()
print("ADDITIONAL ANALYSIS: Comparing Multiple Models")

#Reload data for comparison, same idea as above, except now we're using sklearn and the .ravel() method will be needed
#to flatten the 2D NumPy array returned by selecting the 'total_earnings' column into a 1D array. When a single
#column is extracted from a DataFrame using double brackets like df[['total_earnings']] as seen above, it preserves
#the 2D structure with shape (n, 1). But, sklearn's regression models are expecting the target variable to be a 1D
#array of shape (n). So, using .ravel() allows compatability by converting the array into the correct shape without
#having to make unnecessary copies of the data. Definitely a new method I learned this week that's pretty cool
X_train = training_data_df.drop('total_earnings', axis=1).values
Y_train = training_data_df['total_earnings'].values.ravel()
X_test = test_data_df.drop('total_earnings', axis=1).values
Y_test = test_data_df['total_earnings'].values.ravel()

#Here's sklearn's LinearRegression model, just a simple LR model on the scaled training and test datasets. First,
#a LinearRegression object is instantiated, which models the relationship between the input features and the target
#variable using a best-fit line. The model is trained using the fit() method same as above, then generates predictions
#on test features using predict(). Once again we're using MSE to assess performance between the predicted and actual
#test set values, and calculating the error using mean_squared_error(). The result is printed to output using 6 decimal
#places of precision, allowing direct comparison with other models later in the analysis
lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)
lr_predictions = lin_model.predict(X_test)
lr_mse = mean_squared_error(Y_test, lr_predictions)
print("Linear Regression MSE on test set: {:.6f}".format(lr_mse))

#Here's sklearn's DecisionTreeRegressor to model the relationship between the input features and target earnings.
#This decision tree works by recursively splitting the data into regions based on feature thresholds that minimize
#prediction error. We set random_state=13 (my lucky number) to ensure reproducibility, so that the same splits and
#structures are produced on each run. After fitting the model to the training data, it generates predictions on the
#test set. MSE is called again to evaluate the performance, the result is saved into tree_mse and printed to output
tree_model = DecisionTreeRegressor(random_state=13)
tree_model.fit(X_train, Y_train)
tree_predictions = tree_model.predict(X_test)
tree_mse = mean_squared_error(Y_test, tree_predictions)
print("Decision Tree Regressor MSE on test set: {:.6f}".format(tree_mse))

#Here's sklearn's RandomForrestRegressor to evaluate how an ensemble of decision trees performs on the test set. A
#random forrest combines predictions from multiple trees (in this case, 100) in an attempt to reduce overfitting and
#improve generalization. What's interesting about a random forrest ensemble is how it introduces randomness both in
#the data and in the features. Each individual tree is trained on a different bootstrap sample, a randomly drawn
#subset of the training data which ensures that no two trees see exactly the same examples. At each decision node,
#the model considers only a random subset of the available features when choosing the best split This prevents dominant
#features from overwhelming the model and encourages diversity across the trees. Once all trees are trained, their
#individual predictions are averaged to produce the final output, which tends to be more accurate and stable than any
#single individual tree. Ensemble approaches like this are great at mitigating the high variance typically associated
#with decision trees, making random forests a stronger and more versatile model, especially with nonlinear relationships
#or noisy datasets
rf_model = RandomForestRegressor(random_state=13, n_estimators=100)
rf_model.fit(X_train, Y_train)
rf_predictions = rf_model.predict(X_test)
rf_mse = mean_squared_error(Y_test, rf_predictions)
print("Random Forest Regressor MSE on test set: {:.6f}".format(rf_mse))

#This line just stores the MSE previously calculated during the evaluation phase of step 3. We could honestly remove
#this line since our test_error_rate is already holding this value, but we'll keep it here for readability and
#modularity purposes
nn_mse = test_error_rate

#We'll make a dictionary to store the model names as keys and their corresponding MSE values as the dictionary's values
results = {
    "Linear Regression": lr_mse,
    "Decision Tree": tree_mse,
    "Random Forest": rf_mse,
    "Neural Network": nn_mse
}

#Here we sort the dictionary of model performance results by MSE in ascending order so that the best performing models
#appear first. The sorted() function is applied to the .items() of the dictionary, which gives a sequence of
#(model_name, mse_value) tuples. We want to make sure that the sorting is based on the MSE values, to do that, we will
#use a 'key' argument using a lambda function: lambda x: x[1]. You may be asking why in the world do we need to use a
#lambda function here? This anonymous function tells Python to sort each tuple based on its 2nd element, the error
#value. If we didn't use a key=lambda approach Python would sort the tuples based on their first element by default,
#which in this case is the model name in alphabetical order. By explicitly telling sorted() to look at x[1], we can
#make sure that the models are ranked according to their actual performance on the test set. Once sorted, the loop
#iterates through each tuple and prints the model name along with its MSE, formatted to six decimal places for easy
#comparison. This will quickly show us which regression method generalizes best to unseen data. Other approaches
#might define a sorting function like say sort_by_mse(item): return item[1]; sorted(results.items(), key=sort_by_mse)
#but I think just using lambda is probably more elegant if not a bit esoteric at first glance. We're here to learn
sorted_results = sorted(results.items(), key=lambda x: x[1])
print("\nModel Performance Summary (Lowest MSE first):")
for name, mse in sorted_results:
    print("{}: {:.6f}".format(name, mse))


#Heading into our visualization pipeline here, this block extracts model names and their corresponding MSE values
#so they can be passed to the bar chart. List comprehensions separate the tuples into two aligned lists: model_names
#for the x-axis label and mse_values for the y-axis heights of the bars. We're using the _ throwaway variable to ignore
#the part of the tuple that we don't need for each axis. This approach keeps the model names and values in the correct
#order for plotting
model_names = [name for name, _ in sorted_results]
mse_values = [mse for _, mse in sorted_results]

#PyPlot initialization to compare model performance based on MSE. Basic bar chart with solid ('-') horizontal grid
#lines that are 70% opaque
plt.figure(figsize=(10, 6))
bars = plt.bar(model_names, mse_values)
plt.xlabel("Model")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Model Comparison - MSE on Test Set")
plt.grid(axis='y', linestyle='-', alpha=0.7)

#Loops through the bars in the plot and adds numeric labels on top of each bar to make the MSE values easier to
#interpret at a glance. The plt.text() function then places the value slightly above the top of the bar by offsetting
#the y-coordinate with a small constant. The x-coordinate is set to the center of the bar using the bar's x-position,
#plus half of its width. The label is horizontally centered (ha='center'), vertically aligned to the bottom
#(va='bottom'), and rendered with a medium font size of nine. Leads to a simple and clean chart in my opinion
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.00005,
             "{:.6f}".format(height), ha='center', va='bottom', fontsize=9)

#These two lines render the chart. Using plt.tight_layout() adjusts the spacing of plot elements to avoid overlap
plt.tight_layout()
plt.show()

#We can see from the previous plot that our RandomForrestRegressor performed best by a significant margin. So, we'll
#use what it learned to analyze which input features had the most influence on its predictions. What's cool about
#random forests is that they naturally provide a measure of feature importance by evaluating how much each feature
#reduces prediction error when splitting nodes across all trees in the ensemble. We can visualize this insight.
#We'll start the setup by extracting the feature names and calculating the importance scores assigned by the trained
#model, which we can then sort and display in text and in a PyPlot
print("\nFeature Importance Analysis:")
feature_names = training_data_df.drop('total_earnings', axis=1).columns

#We begin the process of interpreting the trained RandomForrestRegressor by analyzing which features contributed most
#to the model's decisions. These values are stored in feature_importances_ You might be wondering, why is there an
#extra underscore after feature_importances_? In sklearn, attributes that are learned or computed during training are
#given a trailing underscore to distinguish them from parameters that are set manually when creating the model. So, the
#extra underscore here means that 'this value didn't exist until after .fit() was called' and it signals that the
#attribute is derived from the data, not something that was passed in manually. To sort importance scores from highest
#to lowest, the np.argsort(rf_importance) function returns the indices that would sort the array in ascending order.
#Features with the lowest importance would hence come first. But we want to rank the features the other way around.
#So, we can reverse that list using slicing syntax [::-1] and the final result, rf_indices, ends up as an array of
#indices that correspond to the features in descending order of importance. This allows us to reference the original
#feature_names list and output the top-ranked features.
rf_importance = rf_model.feature_importances_
rf_indices = np.argsort(rf_importance)[::-1]

#Now we can print the names and importance scores of the top five most influential features in the trained
#RandomForrestRegressor. The range(min(5, len(feature_names))) loop will run only as many times as we have features.
#This method would work even if we only had four features to analyze but would return at most five. Inside the loop,
#we can use the precomputed rf_indices array to access the features in descending order of importance. The print()
#statement here uses .format() to display each feature name alongside its importance score, rounded to four decimals
print("\nRandom Forest - Top 5 Most Important Features:")
for i in range(min(5, len(feature_names))):
    print("{}: {:.4f}".format(feature_names[rf_indices[i]], rf_importance[rf_indices[i]]))

#We'll instantiate another PyPlot to visualize feature importance. Similarly to the MSE plot setup, except this time
#we're using plt.xticks(rotation=45, ha='right') because feature_names can be quite long, giving them an angular
#offset prevents them from overlapping while making them still clearly readable
plt.figure(figsize=(10, 6))
plt.bar(feature_names, rf_importance)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()