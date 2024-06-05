"""
Research project for Msc. Forensic Science

Author: Luca Nederhorst
Academic year: 2023 - 2024

https://www.youtube.com/watch?v=CQveSaMyEwM
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sn
from sklearn.metrics import confusion_matrix

# load data into pandas dataframa
metadata_df = pd.read_csv('metadata_model.csv')
processed_metadata_df = metadata_df


def determine_species(folder_path):
    if 'Chrysomya Albiceps' in folder_path:
        return 'Chrysomya Albiceps'
    elif 'Synthesiomyia Nudiseta' in folder_path:
        return 'Synthesiomyia Nudiseta'
    else:
        return 'Unknown'
    
def reference_number_species(species):
    if 'Chrysomya Albiceps' in species:
        return 1
    elif 'Synthesiomyia Nudiseta' in species:
        return 2
    else:
        return 'Unknown'

def remove_brackets(cel_value):
    cel_value = cel_value.strip('{}')
    return cel_value

def is_number(cel_value):
    try:
        float(cel_value)
        return True
    except ValueError:
        return False


# remove {} from cel value and put back string or number
infinity_indices = []
for column in processed_metadata_df.columns:
    for row_index, value in processed_metadata_df[column].items():
        value_no_brackets = remove_brackets(value)

        if is_number(value_no_brackets) == True:
            value_no_brackets = float(value_no_brackets)
            processed_metadata_df.at[row_index, column] = value_no_brackets

            if np.isinf(value_no_brackets).any():
                print(f'Infinity found in column: {column}, row:{row_index}')
                infinity_indices.append(row_index)
                
        else:
            processed_metadata_df.at[row_index, column] = value_no_brackets

if infinity_indices:
    processed_metadata_df = processed_metadata_df.drop(index=infinity_indices)


# check if there are missing values
missing_value_found = False
for column in processed_metadata_df.columns:
    for row_index, value in processed_metadata_df[column].items():
        if pd.isna(value):
            print(f'Missing value at row {row_index}, column {column}')
            missing_value_found = True

if missing_value_found == False:
    print('No missing values found.')
else:
    print('Missing value(s) found.')


# add column with species and add column with specific reference number of class
processed_metadata_df['species'] = processed_metadata_df['augment_specie_folder_path'].apply(determine_species)
processed_metadata_df['class'] = processed_metadata_df['species'].apply(reference_number_species)
#print(processed_metadata_df['class'].value_counts()) # see amount of datapoints per class

# check distribution of classes
chrysomya = processed_metadata_df[processed_metadata_df['class'] == 1]
synthesiomyia = processed_metadata_df[processed_metadata_df['class'] == 2]

# check the distribution of certain features
plt.scatter(x=chrysomya['homogeneity'], y=chrysomya['contrast'], color='blue', label='Chrysomya')
plt.scatter(x=synthesiomyia['homogeneity'], y=synthesiomyia['contrast'], color='red', label='Synthesiomyia')
plt.xlabel('Homogeneity')
plt.ylabel('Contrast')
plt.title('Scatter Plot')
plt.legend()
#plt.show()

# check if there are unwanted columns
# if non-numeric remove
processed_metadata_df = processed_metadata_df.apply(pd.to_numeric, errors='coerce')
processed_metadata_df = processed_metadata_df.dropna(axis=1, how='all')

# save processed data to new csv file
processed_metadata_df.to_csv('processed_metadata.csv', index=False)

# create independent variable (feature dataframa)
feature_df = processed_metadata_df[['area', 'perimeter', 'circularity_ratio', 'eccentricity',
       'major_axis_length', 'minor_axis_length', 'convex_area', 'solidity',
       'equivalent_diameter_area', 'spatial_frequency_1',
       'spatial_frequency_2', 'hu_moment_1', 'hu_moment_2', 'hu_moment_3',
       'hu_moment_4', 'hu_moment_5', 'hu_moment_6', 'hu_moment_7', 'contrast',
       'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM',
       'mean_hue_hsv', 'std_hue_hsv', 'mean_sat_hsv', 'std_sat_hsv',
       'mean_hue_LCH', 'std_hue_LCH', 'mean_sat_LCH', 'std_sat_LCH',
       'mean_luminance', 'std_sat_lab']]

independent_variable_x = np.asarray(feature_df)

# create dependent variable
class_df = processed_metadata_df['class']
dependent_variable_y = np.asarray(class_df)

# divide data in train and test dataset
x_train, x_test, y_train, y_test = train_test_split(independent_variable_x, 
                                                    dependent_variable_y, 
                                                    test_size=0.2, 
                                                    random_state=4)

# KNN modelling and evaluation, rest is in default, metric is minkowski is default, meaning euclidean distance
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(x_train, y_train)
#knn_classifier.score(x_test, y_test)
y_predict = knn_classifier.predict(x_test)
report = classification_report(y_test, y_predict)
print(report)

# print out grid to see what is predicted correctly and not
confusion_matrix = confusion_matrix(y_test, y_predict)
plt.figure(figsize=(7,5))
sn.heatmap(confusion_matrix, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()