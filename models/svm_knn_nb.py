"""
Module: Training of SVM, KNN and NB

This module contains training of SVM, KNN and NB based on manually extracted features from photos (Cloud and ML Online, 2019).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time

sns.set_style('darkgrid')


def determine_species(folder_path):
    """ Function to determine species out of folder path """

    if 'Chrysomya albiceps adult' in folder_path:
        return 'Chrysomya albiceps adult'
    elif 'Synthesiomyia nudiseta adult' in folder_path:
        return 'Synthesiomyia nudiseta adult'
    elif 'Chrysomya megacephala adult' in folder_path:
        return 'Chrysomya megacephala adult'
    elif 'Chrysomya albiceps larvae' in folder_path:
        return 'Chrysomya albiceps larvae'
    elif 'Synthesiomyia nudiseta larvae' in folder_path:
        return 'Synthesiomyia nudiseta larvae'


def reference_number_species(species):
    """ Create reference number for certain species """
    
    if 'Chrysomya albiceps adult' in species:
        return 1
    elif 'Synthesiomyia nudiseta adult' in species:
        return 2
    elif 'Chrysomya megacephala adult' in species:
        return 3
    elif 'Chrysomya albiceps larvae' in species:
        return 4
    elif 'Synthesiomyia nudiseta larvae' in species:
        return 5


def remove_brackets(cel_value):
    """ Remove all brackets within cells in metadata file """

    cel_value = cel_value.strip('{}')
    return cel_value


def is_number(cel_value):
    """ Set cel value to floating number if possible """
    try:
        float(cel_value)
        return True
    except ValueError:
        return False


# load data into pandas dataframa
METADATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'annotation_files', 'metadata_model.csv')
metadata_df = pd.read_csv(METADATA_PATH)
processed_metadata_df = metadata_df

# remove {} from cel value and put back string or number
infinity_indices = []
for column in processed_metadata_df.columns:
    for row_index, value in processed_metadata_df[column].items():
        value_no_brackets = remove_brackets(value)

        if is_number(value_no_brackets) is True:
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

if missing_value_found is False:
    print('No missing values found.')
else:
    print('Missing value(s) found.')


# add column with species and add column with specific reference number of class
processed_metadata_df['species'] = processed_metadata_df['augment_specie_folder_path'].apply(determine_species)
processed_metadata_df['class'] = processed_metadata_df['species'].apply(reference_number_species)
print(processed_metadata_df['class'].value_counts())    # see amount of datapoints per class

# check if there are unwanted non-numeric columns
processed_metadata_df = processed_metadata_df.apply(pd.to_numeric, errors='coerce')
processed_metadata_df = processed_metadata_df.dropna(axis=1, how='all')

# save processed data to new csv file
PROCESSED_PATH = os.path.join(os.path.dirname(__file__), '..', 'annotation_files', 'processed_metadata.csv')
processed_metadata_df.to_csv(PROCESSED_PATH, index=False)

# create independent variable (feature dataframa)
feature_df = processed_metadata_df[['area', 'perimeter', 'circularity_ratio',
                                    'eccentricity', 'major_axis_length',
                                    'minor_axis_length', 'convex_area',
                                    'solidity', 'equivalent_diameter_area',
                                    'spatial_frequency_1', 'spatial_frequency_2',
                                    'hu_moment_1', 'hu_moment_2', 'hu_moment_3',
                                    'hu_moment_4', 'hu_moment_5', 'hu_moment_6',
                                    'hu_moment_7', 'contrast', 'dissimilarity',
                                    'homogeneity', 'energy', 'correlation', 'ASM',
                                    'mean_hue_hsv', 'std_hue_hsv', 'mean_sat_hsv',
                                    'std_sat_hsv', 'mean_hue_LCH', 'std_hue_LCH',
                                    'mean_sat_LCH', 'std_sat_LCH',
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

# svm training
svm_classifier = svm.SVC(kernel='linear', gamma='auto', C=1)

start_time_svm = time.time()
svm_classifier.fit(x_train, y_train)
end_time_svm = time.time()
train_time_svm = end_time_svm - start_time_svm
print(f'Training time SVM: {train_time_svm}')

y_predict_svm = svm_classifier.predict(x_test)

# svm performance evaluation
accuracy_svm = accuracy_score(y_test, y_predict_svm)
report = classification_report(y_test, y_predict_svm)
print(f'Accuracy: {accuracy_svm}')
print('Classification Report SVM:')
print(report)

confusion_matrix_svm = confusion_matrix(y_test, y_predict_svm)
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix_svm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix SVM')
plt.show()

dump(svm_classifier, os.path.join(os.path.dirname(__file__), '..', 'trained_models', 'InClasSVM_Model.joblib'))

# knn training using euclidean distance
knn_classifier = KNeighborsClassifier(n_neighbors=5)

start_time_knn = time.time()
knn_classifier.fit(x_train, y_train)
end_time_knn = time.time()
train_time_knn = end_time_knn - start_time_knn
print(f'Training time KNN: {train_time_knn}')

y_predict_knn = knn_classifier.predict(x_test)

# knn performance evaluation
accuracy_knn = accuracy_score(y_test, y_predict_knn)
report = classification_report(y_test, y_predict_knn)
print(f'Accuracy: {accuracy_knn}')
print('Classification Report KNN:')
print(report)

confusion_matrix_knn = confusion_matrix(y_test, y_predict_knn)
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix_knn, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix KNN')
plt.show()

dump(knn_classifier, os.path.join(os.path.dirname(__file__), '..', 'trained_models', 'InClasKNN_Model.joblib'))

# nb training
nb_classifier = GaussianNB()

start_time_nb = time.time()
nb_classifier.fit(x_train, y_train)
end_time_nb = time.time()
train_time_nb = end_time_nb - start_time_nb
print(f'Training time NB: {train_time_nb}')

y_predict_nb = nb_classifier.predict(x_test)

# nb performance evaluation
accuracy_nb = accuracy_score(y_test, y_predict_nb)
report = classification_report(y_test, y_predict_nb)
print(f'Accuracy: {accuracy_nb}')
print('Classification Report NB:')
print(report)

confusion_matrix_nb = confusion_matrix(y_test, y_predict_nb)
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix_nb, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix KNN')
plt.show()

# put accuracies of trained svm, knn and nb in one plot
accuracy_scores = {
    'SVM': accuracy_svm,
    'KNN': accuracy_knn,
    'NB': accuracy_nb
}
plt.figure(figsize=(10, 6))
sns.barplot(x=list(accuracy_scores.keys()), y=list(accuracy_scores.values()))
plt.title('Accuracy different models')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()

dump(nb_classifier, os.path.join(os.path.dirname(__file__), '..', 'trained_models', 'InClasNB_Model.joblib'))
