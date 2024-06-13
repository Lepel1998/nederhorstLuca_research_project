"""
CNN with transfer learning (pretrained model)

Because of small dataset it is important to choose the pretrained model which is good in fine-tuning on small dataset
EfficientNetB0 is chosen for balance between performance and efficiency
Augmentation to artificially increase the dataset size and variability
Freezig layers: initially pre-trained layers are frozen to preserve their learned features
Fine-tuning: unfreeze the last few layers of the pre-trained model and retrain with a lower learning rate to improve performance
https://www.youtube.com/watch?v=fCtMf6qHtdk


"""
import torch
from tensorflow.python.keras.models import load_model
import os
import time
import torch.nn as nn
from torchvision import models
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader, random_split
from pillow_heif import register_heif_opener
register_heif_opener()
sns.set_style('darkgrid')

# load dataset
dataset_directory = 'processed_dataset'

# do augmentation and preprocessing here to skip slow main.py function
preprocessing = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), # converts image to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])    # normalizes tensor using mean and standard deviation values from ImageNet dataset
])

# load dataset 
dataset = datasets.ImageFolder(root = dataset_directory, transform=preprocessing)

# check which label referres to what species
class_labels = dataset.classes
label_to_species = {}
for label in range(len(class_labels)):
    label_to_species[label] = class_labels[label]
print("Label to species:")
print(label_to_species)

# divide dataset into training and test set
train_size = int(0.8*len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# dataloader to load data into differnet batches
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size = 32, shuffle=False)

# load the EfficientNetB0 model retrieved from torch (CNN with transfer learning)
#  "https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth"
model = models.efficientnet_b0(pretrained=True)

# freeze base model, this is the transfer learning part
for parameter in model.parameters():
    parameter.requires_grad = False

# Add custom layers on top of base model, this is also the transfer leanring part
number_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Linear(number_features, 1024),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(1024, len(dataset.classes)),
    nn.Softmax(dim=1)
)

# define loss and optimizer
loss_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4)

# save the history of the training model 
train_losses = []
train_accuracies = []
epoch_amount = 10

# training loop
start_time_cnn = time.time()
for epoch in range(epoch_amount):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predictions = torch.max(outputs, 1)
        running_loss = running_loss + loss.item() * inputs.size(0)
        running_corrects = running_corrects + torch.sum(predictions == labels.data)

    epoch_loss = running_loss / train_size
    epoch_accuracy = running_corrects.double() / train_size 

    # updates about process
    print(f'Epoch {epoch} / {epoch_amount - 1}, loss: {epoch_loss:.4f}, accuracy: {epoch_accuracy:.4f}') 

    # add history to lists
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

"""
You can add fine tuning here

"""
for param in model.features[-20:].parameters():
    param.requires_grad = True

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

fine_tune_epochs = 2
total_epochs = epoch_amount + fine_tune_epochs
for epoch in range(epoch_amount, total_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_dataloader:
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predictions = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(predictions == labels.data)

    epoch_loss = running_loss / train_size
    epoch_acc = running_corrects.double() / train_size

    print(f"Epoch {epoch}/{total_epochs-1}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

    # Append to history lists
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

end_time_cnn = time.time()
train_time_cnn = end_time_cnn - start_time_cnn
print(f'Training time CNN: {train_time_cnn}')

# evaluate the model
model.eval()
validation_loss = 0.0
validation_corrects = 0.0

# save the history of the training model
true_labels = []
predicted_labels = []

with torch.no_grad():
    for inputs, labels in test_dataloader:
        outputs = model(inputs)
        loss = loss_criterion(outputs, labels)

        _, predictions = torch.max(outputs, 1)
        validation_loss = validation_loss + loss.item() * inputs.size(0)
        validation_corrects = validation_corrects + torch.sum(predictions == labels.data)

        true_labels.extend(labels.cpu().numpy().astype(int))
        predicted_labels.extend(predictions.cpu().numpy().astype(int))
        #predicted_labels.append(validation_corrects)
       
print(f'True labels: {true_labels[:5]}')
print(f'Predicted labels: {predicted_labels[:5]}')   
validation_loss = validation_loss / test_size
validation_accuracy = validation_corrects.double() / test_size

print(f'Validation loss: {validation_loss:.4f}, Validation Accuracy: {validation_accuracy:.4f}')

# plot accuracy of model training in plot
plt.plot(train_losses, 'r', label='train_loss')
plt.plot(true_labels, 'm', label='validation_loss')
plt.plot(train_accuracies, 'b', label='train_accuracy')
plt.plot(predicted_labels, 'g', label='validation accuracy')
plt.suptitle('Loss and Accuracy', fontsize=22)
plt.legend(loc='right')
plt.ylabel('Value')
plt.ylim(0, 1)
plt.show()

# print 
report = classification_report(true_labels, predicted_labels)
print('\nClassification Report CNN:')
print(report)

# confusion matrix
confusion_matrix_pretrained_cnn = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(10,7))
sns.heatmap(confusion_matrix_pretrained_cnn, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix CNN')
plt.show()


model_path = 'InClasPreTrainedCNN_Model.h5'
torch.save(model.state_dict(), model_path)
print(f'Model saved to {model_path}')

