"""
    Naive Bayes algorithm
    https://www.youtube.com/watch?v=3I8oX3OUL6I

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
sns.set_style('darkgrid')
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report



# load data
processed_metadata_df = pd.read_csv('processed_metadata.csv')
processed_metadata_df['class'].hist()
plt.show()

# check if features are correlated
corr = processed_metadata_df.iloc[:,:-1].corr(method="pearson")
cmap = sns.diverging_palette(250, 354, 80, 60, center='dark', as_cmap=True)
sns.heatmap(corr, vmax=1, vmin=-0.5, cmap=cmap, square=True, linewidths=0.2)
plt.show()


# multinomial, bernoulli, gaussian
# we will go for gaussian as this is suitable for continuous values
# check if all features are normally distributed with shapiro-wilk test
#for column in processed_metadata_df:
#    if processed_metadata_df[column].dtype in ['float64', 'int64']:
#        processed_metadata_df[column] = np.log1p(processed_metadata_df[column])
#        stat, p = shapiro(processed_metadata_df[column])
#        print(f'Shapiro-Wilk test for {column}: stat={stat}, p-value={p}')


# split dataset into train and test
independent_variable_x = processed_metadata_df.drop('class', axis=1)
dependent_variable_y = processed_metadata_df['class']

# split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(independent_variable_x, 
                                                    dependent_variable_y, 
                                                    test_size=0.2, 
                                                    random_state=4)
# initiate Gaussian Naive Bayes classifier
nb_classifier = GaussianNB()

# fit model on training data
nb_classifier.fit(x_train, y_train)
y_predict = nb_classifier.predict(x_test)

# evaluate accuracy
accuracy = accuracy_score(y_test, y_predict)
report = classification_report(y_test, y_predict)
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)
