"""
Module: Convolutional Neural Network using Tensorflow Keras API

This module contains the set-up of the model using Keras API.
Leading code followed https://www.youtube.com/watch?v=jztwpsIzEGc&t=300s
"""
import os
import time
import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.python.keras.metrics import Precision, Recall, CategoricalAccuracy
from tensorflow.python.keras.models import Sequential, load_model

# set environment variable to disable optimizations (avoid compatibiliy issues)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# set seaborn style for plots
sns.set_style('darkgrid')


# check if dataset is distributed and assign to adapter
def _is_distributed_dataset(dataset):
    return isinstance(dataset, data_adapter.input_lib.DistributedDatasetSpec)


data_adapter._is_distributed_dataset = _is_distributed_dataset


# load data and create iterator
IMAGE_DATASET_PATH = 'processed_dataset'
image_dataset = tf.keras.utils.image_dataset_from_directory(IMAGE_DATASET_PATH,
                                                            batch_size=32,
                                                            image_size=(227, 227))
dataset_iterator = image_dataset.as_numpy_iterator()
batch = dataset_iterator.next()

# preprocess data to scale values between 0 and 1 per pixel
process_image_dataset = image_dataset.map(lambda x, y: (x/227, y))
scaled_iterator = process_image_dataset.as_numpy_iterator()
batch = scaled_iterator.next()

# split data in train, validate and test datasets
train_size = int(len(process_image_dataset)*0.7)
train_dataset = process_image_dataset.take(train_size)
validate_size = int(len(process_image_dataset)*0.2)
validate_dataset = process_image_dataset.skip(train_size).take(validate_size)
test_size = int(len(process_image_dataset)*0.1)
test_dataset = process_image_dataset.skip(train_size + validate_size).take(test_size)

# build convolutional neural network using Sequential from Tensorflow Keras APi
model = Sequential([
    Conv2D(16, (3, 3), 1, activation='relu', input_shape=(227, 227, 3)),
    Conv2D(16, (3, 3), 1, activation='relu'),
    MaxPooling2D(),
    Conv2D(32, (3, 3), 1, activation='relu'),
    MaxPooling2D(),
    Conv2D(16, (3, 3), 1, activation='relu'),
    Conv2D(16, (3, 3), 1, activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(5, activation='softmax')
])

# compile model with optimizer, loss function and metrics
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# training of model, visualize performace using TensorBoard, and save process
current_directory = os.getcwd()
LOG_PROCESS = os.path.join(current_directory, 'log_process_CNN')
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_PROCESS)

start_time_cnn = time.time()
history = model.fit(train_dataset,
                    epochs=20,
                    validation_data=validate_dataset,
                    callbacks=[tb_callback])
end_time_cnn = time.time()
train_time_cnn = end_time_cnn - start_time_cnn
print(f'Training time CNN: {train_time_cnn}')

# save trainedmodel
model.save(os.path.join('trained_models', 'InClasCNN_Model.h5'))
new_model = load_model(os.path.join('trained_models', 'InClasCNN_Model.h5'))

# plot training history: loss an accuracy
plt.figure()
plt.plot(history.history['loss'], 'r', label='loss')
plt.plot(history.history['val_loss'], 'm', label='val_loss')
plt.plot(history.history['accuracy'], 'b', label='accuracy')
plt.plot(history.history['val_accuracy'], 'g', label='val_accuracy')
plt.suptitle('Loss and Accuracy', fontsize=20)
plt.legend(loc="upper right")
plt.show()

# evaluation of model performance with test dataset
accuracy = CategoricalAccuracy()
precision = Precision()
recall = Recall()
CLASS_AMOUNT = 5
true_labels = []
predicted_labels = []

for batch in test_dataset.as_numpy_iterator():
    # extract image and label from batch
    image, label = batch

    # check label dimensions and convert if necessary
    if label.ndim > 1 and label.shape[-1] > 1:
        label = np.argmax(label, axis=1)

    # predict label probabilities using model
    predicted_label = model.predict(image)

    # get predicted class
    predicted_class = np.argmax(predicted_label, axis=1)
    print(f'Predicted class:{predicted_class}')

    # convert predicted class to one-hot-encoding
    predicted_class_hot_label = tf.keras.utils.to_categorical(predicted_class,
                                                              num_classes=CLASS_AMOUNT)

    # convert true label to one-hot encoding
    true_labels_hot_label = tf.keras.utils.to_categorical(label,
                                                          num_classes=CLASS_AMOUNT)

    # update accurcay, precision, recall metrics based on batch analysis
    accuracy.update_state(true_labels_hot_label,
                          predicted_class_hot_label)
    precision.update_state(true_labels_hot_label,
                           predicted_class_hot_label)
    recall.update_state(true_labels_hot_label,
                        predicted_class_hot_label)

    # extend with current batch to accumulate for evaluation
    true_labels.extend(label)
    predicted_labels.extend(predicted_class)

# print overall precision, accuracy and recall of model
precision = precision.result().numpy()
accuracy = accuracy.result().numpy()
recall = recall.result().numpy()
print(f'Precision: {precision}, Accuracy: {accuracy}, Recall: {recall}')

true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

# generate classification report
report = classification_report(true_labels,
                               predicted_labels)
print('\nClassification Report CNN:')
print(report)

# generate confusion matrix
confusion_matrix_cnn = confusion_matrix(true_labels,
                                        predicted_labels)
print('Confusion Matrix CNN:')
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix_cnn, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix CNN')
plt.show()
