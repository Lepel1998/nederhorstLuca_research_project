import pip
import numpy as np
import pandas as pd
import matplotlib as plt
import os


path = "C:/Users/luca-/Documents/Forensic Science/year 2/Research/Research Project/archive/test/test"
bus_types = os.listdir(path)
#print(bus_types)
#print ('Types of busses found:', len(bus_types))

busses = []

for item in bus_types:
    # get all the file names
    all_busses = os.listdir(path + '/' + item)
    #print(all_busses)

    for bus in all_busses:
        busses.append((item, str(path + '/' + item) + '/' + bus))

# build a dataframe with columns of both electric car and electric bus
busses_df = pd.DataFrame(data=busses, columns=['bus type', 'image'])
#print(busses_df.head())
#print(busses_df.tail())

# Let's check how many samples for each category are present
print("Total number of electrics vehicles in the dataset:", len(busses_df))

busses_count = busses_df['bus type'].value_counts()
print("busses in each category")
print(busses_count)


# very important thing in image classification will be that you need to resize the image to a particular shape
import cv2
path = "C:/Users/luca-/Documents/Forensic Science/year 2/Research/Research Project/archive/test/test/"

im_size = 60

images = []
labels = []

for i in bus_types:
    data_path = path + str(i)   # entered in 1st folder and then 2nd folder
    filenames = [i for i in os.listdir(data_path)]
    for f in filenames:
        img = cv2.imread(data_path + '/' + f)
        img = cv2.resize(img, (im_size, im_size))
        images.append(img)
        labels.append(i)

#print(images)
#print(labels)


# transform the image array to a numpy type
images = np.array(images)
images.shape
print(images.shape)

images = images.astype('float32') / 255
print(images.shape)
 # now all the preprocessing is all set for the object classification

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

y = busses_df['bus type'].values

# for y
y_labelencoder = LabelEncoder ()
y = y_labelencoder.fit_transform (y)
#print(y)

 # one hot encoding means that you transform categorial data into numerical data that is better usable for machine learning goals
y = y.reshape(-1,1)

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('my_ohe', OneHotEncoder(), [0])], remainder = 'passthrough')
Y = ct.fit_transform(y)#.toarray()
print(Y.shape)

# gebleven bij minuut 12.37 in video https://www.youtube.com/watch?v=JQWCAgwpXbo&t=165s

















