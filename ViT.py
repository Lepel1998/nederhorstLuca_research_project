import time
import numpy as np
import torch
from transformers import ViTFeatureExtractor, ViTModel
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('darkgrid')


# load pre-trained ViT model and feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k', do_rescale=False)
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
print('pretrained model retrieved')

# load dataset
dataset_path = 'dataset'
dataset = ImageFolder(root=dataset_path, transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
]))

# define DataLoader
batch_size = 8
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# extract features from images
features = []
labels = []

for images, labels_batch in dataloader:
    inputs = feature_extractor(images=images, return_tensors="pt")
    outputs = model(**inputs)
    features.append(outputs.pooler_output)
    labels.extend(labels_batch)
print('all features extracted through pretrained model')

# concatenate features and convert labels to tensor
features = torch.cat(features, dim=0)
labels = torch.tensor(labels)

# split data into train (80%) and test (20%) sets randomly
train_features, test_features, train_labels, test_labels = train_test_split(features, 
                                                                            labels, 
                                                                            test_size=0.2, 
                                                                            random_state=2, 
                                                                            stratify=labels)
print(train_labels, test_labels)

# convert tensors to NumPy arrays, because Scikit-Learn  classifiers require inputs in the form of numpy arrays
train_features = train_features.detach().cpu().numpy()
test_features = test_features.detach().cpu().numpy()
train_labels = train_labels.detach().cpu().numpy()
test_labels = test_labels.detach().cpu().numpy()
all_labels = dataset.classes

# train SVM classifier and predict label on features extracted from ViT
svm_classifier = SVC()

start_time_nb = time.time()
svm_classifier.fit(train_features, train_labels)
end_time_nb = time.time()
train_time_nb = end_time_nb - start_time_nb
print(f'Training time SVM: {train_time_nb}')

svm_predictions = svm_classifier.predict(test_features)
svm_accuracy = accuracy_score(test_labels, svm_predictions)
print(f'SVM Accuracy: {svm_accuracy}')

report_svm = classification_report(test_labels, svm_predictions, labels=range(len(all_labels)), target_names=all_labels)
print('\nClassification Report SVM:')
print(report_svm)

confusion_matrix_svm = confusion_matrix(test_labels, svm_predictions, labels=range(len(all_labels)))
print('Confusion Matrix SVM:')
plt.figure(figsize=(10,7))
sns.heatmap(confusion_matrix_svm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix CNN')
plt.show()

# train KNN classifier and predict label on features extracted from ViT
knn_classifier = KNeighborsClassifier()

start_time_nb = time.time()
knn_classifier.fit(train_features, train_labels)
end_time_nb = time.time()
train_time_nb = end_time_nb - start_time_nb
print(f'Training time KNN: {train_time_nb}')

knn_predictions = knn_classifier.predict(test_features)
knn_accuracy = accuracy_score(test_labels, knn_predictions)
print(f'KNN Accuracy: {knn_accuracy}')

report_knn = classification_report(test_labels, knn_predictions, labels=range(len(all_labels)), target_names=all_labels)
print('\nClassification Report KNN:')
print(report_knn)

confusion_matrix_knn = confusion_matrix(test_labels, knn_predictions, labels=range(len(all_labels)))
print('Confusion Matrix KNN:')
plt.figure(figsize=(10,7))
sns.heatmap(confusion_matrix_knn, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix KNN')
plt.show()

# train NB classifier and predict label on features extracted from ViT
nb_classifier = GaussianNB()

start_time_nb = time.time()
nb_classifier.fit(train_features, train_labels)
end_time_nb = time.time()
train_time_nb = end_time_nb - start_time_nb
print(f'Training time NB: {train_time_nb}')

nb_predictions = nb_classifier.predict(test_features)
nb_accuracy = accuracy_score(test_labels, nb_predictions)
print(f'NB Accuracy: {nb_accuracy}')

report_nb = classification_report(test_labels, nb_predictions, labels=range(len(all_labels)), target_names=all_labels)
print('\nClassification Report NB:')
print(report_nb)

confusion_matrix_nb = confusion_matrix(test_labels, nb_predictions, labels=range(len(all_labels)))
print('Confusion Matrix NB:')
plt.figure(figsize=(10,7))
sns.heatmap(confusion_matrix_nb, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix NB')
plt.show()






