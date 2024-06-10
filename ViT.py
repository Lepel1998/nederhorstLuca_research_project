import os
import torch
import requests
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Load pre-trained ViT model and feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k', do_rescale=False)
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

print('pretrained model retrieved')

# Load your dataset
dataset_path = 'processed_dataset'  # Update with your dataset path
dataset = ImageFolder(root=dataset_path, transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
]))

# Define DataLoader
batch_size = 8
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Extract features from images
features = []
labels = []

for images, labels_batch in dataloader:
    inputs = feature_extractor(images=images, return_tensors="pt")
    outputs = model(**inputs)
    features.append(outputs.pooler_output)
    labels.extend(labels_batch)
print('all features extracted through pretrained model')

# Concatenate features and convert labels to tensor
features = torch.cat(features, dim=0)
labels = torch.tensor(labels)
(print('1'))

# Split data into train and test sets
train_features, test_features = features[:int(0.8 * len(features))], features[int(0.8 * len(features)):]
train_labels, test_labels = labels[:int(0.8 * len(labels))], labels[int(0.8 * len(labels)):]

# Convert tensors to NumPy arrays
train_features = train_features.detach().cpu().numpy()
test_features = test_features.detach().cpu().numpy()
train_labels = train_labels.detach().cpu().numpy()
test_labels = test_labels.detach().cpu().numpy()

print('2')


# Train and predict SVM classifier on features extracted from ViT
svm_classifier = SVC()
svm_classifier.fit(train_features, train_labels)
svm_predictions = svm_classifier.predict(test_features)
svm_accuracy = accuracy_score(test_labels, svm_predictions)
print(f'SVM Accuracy: {svm_accuracy}')

print('3')


# Train and predict KNN classifier on features extracted from ViT
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(train_features, train_labels)
knn_predictions = knn_classifier.predict(test_features)
knn_accuracy = accuracy_score(test_labels, knn_predictions)
print(f'KNN Accuracy: {knn_accuracy}')

print('4')


# Train and predict NB classifier on features extracted from ViT
nb_classifier = GaussianNB()
nb_classifier.fit(train_features, train_labels)
nb_predictions = nb_classifier.predict(test_features)
nb_accuracy = accuracy_score(test_labels, nb_predictions)
print(f'NB Accuracy: {nb_accuracy}')

print('5')





