"""
Module: CNN with transfer learning (pretrained model)

Because of small dataset it is important to choose the pretrained model which is good on small datasets.
The pretrained EfficientNetB0 is chosen for balance between performance and efficiency.
Freezing layers: initially pre-trained layers are frozen to preserve their learned features.
Fine-tuning: unfreeze the last few layers of the pre-trained model and retrain with a
lower learning rate to improve performance. Lower learning rate means the new learning will not influence 
the already trained layers a lot!
Pretrained EfficientNetB0 implementation and model (Singla, 2021; Wright, n.d.).
"""

import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from pillow_heif import register_heif_opener
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

register_heif_opener()
sns.set_style('darkgrid')


# path to dataset
DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'processed_dataset')

# normalizes tensor using mean and standard deviation values from ImageNet dataset
# will result in more stable and efficient training process
preprocessing = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load dataset with preprocessing transformations
dataset = datasets.ImageFolder(root=DATASET_PATH,
                               transform=preprocessing)

# map class labels to species names
class_labels = dataset.classes
label_to_species = {}
for label in range(len(class_labels)):
    label_to_species[label] = class_labels[label]
print("Label to species:")
print(label_to_species)

# split dataset into training and testing sets (80% train, 20% test)
train_size = int(0.8*len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# load data in batches for training and testing
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# load EfficientNetB0 model pre-trained on ImageNet dataset
model = models.efficientnet_b0(pretrained=True)

# freeze the pretrained model layers to maintain their learned features
for parameter in model.parameters():
    parameter.requires_grad = False

# add custom layer on top of base model for our specific classification task (transfer learning)
number_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Linear(number_features, 1024),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(1024, len(dataset.classes)),
    nn.Softmax(dim=1)
)

# define loss function and optimizer
loss_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4)

# lists to store training history
train_losses = []
train_accuracies = []

# set the number of epochs
# an epoch is a complete pass through the entire training dataset
EPOCH_AMOUNT = 1

# training loop
start_time_cnn = time.time()
for epoch in range(EPOCH_AMOUNT):
    model.train()
    RUNNING_LOSS = 0.0
    RUNNING_CORRECTS = 0

    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predictions = torch.max(outputs, 1)
        RUNNING_LOSS = RUNNING_LOSS + loss.item() * inputs.size(0)
        RUNNING_CORRECTS = RUNNING_CORRECTS + torch.sum(predictions == labels.data)

    epoch_loss = RUNNING_LOSS / train_size
    epoch_accuracy = RUNNING_CORRECTS.double() / train_size

    # updates process
    print(f'Epoch {epoch} / {EPOCH_AMOUNT - 1}, loss: {epoch_loss:.4f}, accuracy: {epoch_accuracy:.4f}')

    # save history
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

# fine tune model by unfreezing the last few layers
for param in model.features[-20:].parameters():
    param.requires_grad = True

# define a new optimizer for fine-tuning
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

FINE_TUNE_EPOCHS = 1
TOTAL_EPOCHS = EPOCH_AMOUNT + FINE_TUNE_EPOCHS
for epoch in range(EPOCH_AMOUNT, TOTAL_EPOCHS):
    model.train()
    RUNNING_LOSS = 0.0
    RUNNING_CORRECTS = 0

    for inputs, labels in train_dataloader:
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predictions = torch.max(outputs, 1)
        RUNNING_LOSS += loss.item() * inputs.size(0)
        RUNNING_CORRECTS += torch.sum(predictions == labels.data)

    epoch_loss = RUNNING_LOSS / train_size
    epoch_acc = RUNNING_CORRECTS.double() / train_size

    print(f"Epoch {epoch}/{TOTAL_EPOCHS-1}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

    # save history
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

end_time_cnn = time.time()
train_time_cnn = end_time_cnn - start_time_cnn
print(f'Training time CNN: {train_time_cnn}')

# evaluate modek on test dataset
model.eval()
VALIDATION_LOSS = 0.0
VALIDATION_CORRECTS = 0.0

# lists to store true and predicted labels for evaluation
true_labels = []
predicted_labels = []

# disable gardient calculation to save memory and computation
with torch.no_grad():
    for inputs, labels in test_dataloader:
        outputs = model(inputs)

        # calculate loss between predicted and actual labels
        loss = loss_criterion(outputs, labels)

        # get index of maximum value in output tensor
        # corresponds to predicted class
        _, predictions = torch.max(outputs, 1)
        VALIDATION_LOSS = VALIDATION_LOSS + loss.item() * inputs.size(0)
        VALIDATION_CORRECTS = VALIDATION_CORRECTS + torch.sum(predictions == labels.data)

        true_labels.extend(labels.cpu().numpy().astype(int))
        predicted_labels.extend(predictions.cpu().numpy().astype(int))

# print a few true and predicted labels to verify
print(f'True labels: {true_labels[:5]}')
print(f'Predicted labels: {predicted_labels[:5]}')

# calculate validation loss and accuracy
VALIDATION_LOSS = VALIDATION_LOSS / test_size
validation_accuracy = VALIDATION_CORRECTS.double() / test_size

print(f'Validation loss: {VALIDATION_LOSS:.4f}, Validation Accuracy: {validation_accuracy:.4f}')

# plot training loss and accuracy
plt.plot(train_losses, 'r', label='train_loss')
plt.plot(true_labels, 'm', label='validation_loss')
plt.plot(train_accuracies, 'b', label='train_accuracy')
plt.plot(predicted_labels, 'g', label='validation accuracy')
plt.suptitle('Loss and Accuracy', fontsize=22)
plt.legend(loc='right')
plt.ylabel('Value')
plt.ylim(0, 1)
plt.show()

# print classification report
report = classification_report(true_labels, predicted_labels)
print('\nClassification Report CNN:')
print(report)

# plot confusion matrix
confusion_matrix_pretrained_cnn = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix_pretrained_cnn, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix CNN')
plt.show()

# save trained model
model_path = os.path.join(os.path.dirname(__file__), '..', 'trained_models', 'InClasPreTrainedCNN_Model.h5')
torch.save(model.state_dict(), model_path)
print(f'Model saved to {model_path}')
