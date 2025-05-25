# setup
import os
from mlwpy import *
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import (datasets, metrics, model_selection as skms, naive_bayes, neighbors)
import pandas as pd
import seaborn as sns
import memory_profiler, sys
from mlwpy import *
from sklearn import datasets, neighbors, model_selection as skms

# Load Iris dataset
iris = datasets.load_iris()

# Create a DataFrame
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)


# Visualize with seaborn
sns.pairplot(iris_df, hue='species')
plt.show()

sns.pairplot(iris_df, hue= 'target', height=2.7)

#References:
#Fenner, M. E. (2020). Machine learning with Python for everyone. Pearson Education.