###
# Inspect Dataset
###

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics, preprocessing, tree
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.tree import export_graphviz, plot_tree
import pydot
import pickle

df = pd.read_csv('dataset.csv', )
print("*** data is loaded ***")

# Replace NaNs in part of dataset
# df.iloc[:, np.r_[16:18, 36]] = df.iloc[:, np.r_[16:18, 36]].replace(np.nan, 0)
# Drop NaNs in whole dataset
df = df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

# Profiling report for variable understanding
# (If very large dataset, use 'df.sample(n=xxx)' or add 'skiprows=range(1, 100000), nrows=10000)' to pd.read_csv above)
if True:  # False to turn off, True to generate profile report.
    import pandas_profiling
    profile = pandas_profiling.ProfileReport(df, title='Pandas Profiling Report',
                                             html={'style': {'full_width': True}}, minimal=True
                                             )
    profile.to_file(output_file="Profile_report_NewSmallDataset.html")
print(df.dtypes)


# Select columns of interest
X = df.iloc[:, np.r_[1]].values
y = df.iloc[:, 2].values
feature_names = df.columns[np.r_[1]]
target_names = ['Not returned', 'Returned']
print(" the following features are included: \n", ', '.join(feature_names))

target_count = df.iloc[:, 28].value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

target_count.plot(kind='bar', title='Count (target)')


# Class count
count_class_0, count_class_1 = df.iloc[:, 28].value_counts()

# Divide by class
df_class_0 = df[df.iloc[:, 28] == 0]
df_class_1 = df[df.iloc[:, 28] == 1]
df_class_0_under = df_class_0.sample(count_class_1)
df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)

print('Random under-sampling:')
print(df_test_under.iloc[:, 28].value_counts())

df_test_under.iloc[:, 28].value_counts().plot(kind='bar', title='Count (target)')

plt.show()
