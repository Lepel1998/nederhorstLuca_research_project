"""
Module: Convolutional Neural Network using Tensorflow Keras API

This module contains the set-up of the model using Keras API.
Leading code followed https://www.youtube.com/watch?v=jztwpsIzEGc&t=300s 

"""

import cv2
import numpy as np
import os
import seaborn as sns
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import tensorflow.python.keras as keras
from matplotlib import pyplot as plt
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.metrics import Precision, Recall, BinaryAccuracy, CategoricalAccuracy, Accuracy
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.python.keras.models import Sequential
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
sns.set_style('darkgrid')
print(tf.__version__)

def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)

data_adapter._is_distributed_dataset = _is_distributed_dataset
sns.set_style('darkgrid')


# load data and create iterator
image_dataset_path = 'processed_dataset'
image_dataset = tf.keras.utils.image_dataset_from_directory(image_dataset_path, batch_size = 32, image_size = (227, 227))
dataset_iterator = image_dataset.as_numpy_iterator() 
batch = dataset_iterator.next()   

# 0 is Chrysomya Albiceps
# 1 is Synthesomya Nudiseta
#batch[0] is our image
#batch[1] is our lable
#fig, ax = plt.subplots(ncols=4, figsize=(20,20))
#for idx, img in enumerate(batch[0][:4]):
#    ax[idx].imshow(img.astype(int))
#    ax[idx].title.set_text(batch[1][idx])
#plt.show()

# preprocess the data to get values between 0 and 1 per pixel
process_image_dataset = image_dataset.map(lambda x,y: (x/227, y))
scaled_iterator = process_image_dataset.as_numpy_iterator()
batch = scaled_iterator.next()

# split data in train, validate and test data
train_size = int(len(process_image_dataset)*0.7)
train_dataset = process_image_dataset.take(train_size)
validate_size = int(len(process_image_dataset)*0.2)
validate_dataset = process_image_dataset.skip(train_size).take(validate_size)
test_size = int(len(process_image_dataset)*0.1)
test_dataset = process_image_dataset.skip(train_size + validate_size).take(test_size)


# build convolutional neural network using Sequential from Tensorflow Keras APi
model = Sequential([
    Conv2D(16, (3, 3), 1, activation = 'relu', input_shape=(227, 227, 3)),
    Conv2D(16, (3, 3), 1, activation = 'relu'),
    MaxPooling2D(),

    Conv2D(32, (3, 3), 1, activation = 'relu'),
    MaxPooling2D(),

    Conv2D(16, (3, 3), 1, activation = 'relu'),
    Conv2D(16, (3, 3), 1, activation = 'relu'),
    MaxPooling2D(),

    Flatten(),

    Dense(64, activation='relu'),
    Dense(5, activation='softmax') # change sigmud to softmax
])


# adam is optimizer (there are a tons of optimizers)
model.compile(optimizer='adam', 
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy']) 
model.summary()

# training of the model and save process
log_process = "C:/Users/luca-/Documents/Forensic Science/year 2/Research/Research Project/AI/Code/log_process_CNN"
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_process) # you can see the performance of your model while training

start_time_cnn = time.time()
history = model.fit(train_dataset, epochs=20, validation_data=validate_dataset, callbacks=[tb_callback])
end_time_cnn = time.time()
train_time_cnn = end_time_cnn - start_time_cnn
print(f'Training time CNN: {train_time_cnn}')

model.save(os.path.join('InClasCNN_Model.h5')) # serialization of model, taking model and we can store model ask disk
new_model = load_model(os.path.join('InClasCNN_Model.h5'))

# history contains all the data ! plot this data, if you see your loss going down but your validation loss going up, this might indicate overfitting
plt.figure()
plt.plot(history.history['loss'], 'r', label='loss')
plt.plot(history.history['val_loss'],'m', label='val_loss')
plt.plot(history.history['accuracy'], 'b', label='accuracy')
plt.plot(history.history['val_accuracy'],'g', label='val_accuracy')
plt.suptitle('Loss and Accuracy', fontsize=20)
plt.legend(loc="upper right")
plt.show()

# evaluation of model performance with test dataset
accuracy = CategoricalAccuracy()
precision = Precision()
recall = Recall()
class_amount = 5
true_labels = []
predicted_labels = []

for batch in test_dataset.as_numpy_iterator():
    image, label = batch
    print(f'Label: {label}')

    if label.ndim > 1 and label.shape[-1] > 1:
        label = np.argmax(label, axis=1 )
        print(f'Label: {label}')

    predicted_label = model.predict(image)
    print(f'Predicted label probabilities:{predicted_label}')

    predicted_class = np.argmax(predicted_label, axis=1)
    print(f'Predicted class:{predicted_class}')

    predicted_class_hot_label = tf.keras.utils.to_categorical(predicted_class, num_classes=class_amount)
    print(f'One hot encoded predicted class: {predicted_class_hot_label}')

    true_labels_hot_label = tf.keras.utils.to_categorical(label, num_classes=class_amount)
    print(f'One hot encoded true classes: {true_labels_hot_label}')

    accuracy.update_state(true_labels_hot_label, predicted_class_hot_label)
    precision.update_state(true_labels_hot_label, predicted_class_hot_label)
    recall.update_state(true_labels_hot_label, predicted_class_hot_label)

    true_labels.extend(label)
    predicted_labels.extend(predicted_class)


print(f'Precision: {precision.result().numpy()}, Accuracy: {accuracy.result().numpy()}, Recall: {recall.result().numpy()}')

true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)


report = classification_report(true_labels, predicted_labels)
print('\nClassification Report CNN:')
print(report)

confusion_matrix_cnn = confusion_matrix(true_labels, predicted_labels)
print('Confusion Matrix CNN:')
plt.figure(figsize=(10,7))
sns.heatmap(confusion_matrix_cnn, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix CNN')
plt.show()


# test the model with random image from the internet - Chrysomya Albiceps
#random_image = cv2.imread('image.png')
#plt.imshow(cv2.cvtColor(random_image, cv2.COLOR_BGR2RGB))
#resize = tf.image.resize(random_image, (227, 227))
#plt.imshow(resize.numpy().astype(int))


#predicted_label = model.predict(np.expand_dims(resize/227, 0))
#if predicted_label > 0.5:
#    print('Predicted class is Synthesomya Nudiseta')
#else:
#    print('Predicted class is Chrysomya Albiceps')


# save model to working directory













