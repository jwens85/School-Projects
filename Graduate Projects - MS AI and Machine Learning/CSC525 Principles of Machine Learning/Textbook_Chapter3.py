# setup
from mlwpy import *
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd

iris = datasets.load_iris()

iris_df = pd.DataFrame(iris.data, columns= iris.feature_names)
iris_df['target'] = iris.target
print(pd.concat([iris_df.head(3), iris_df.tail(3)]))

#Let's run chapter 3 as a Jupyter notebook to better follow along with the text