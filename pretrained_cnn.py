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
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from pillow_heif import register_heif_opener
register_heif_opener()

# load dataset
dataset_directory = 'adult_dataset'

# do augmentation and preprocessing here to skip slow main.py function
augmentation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # converts image to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])    # normalizes tensor using mean and standard deviation values from ImageNet dataset
])

# load dataset and perform augmentation
dataset = datasets.ImageFolder(root = dataset_directory, transform=augmentation)
train_size = int(0.8*len(dataset))
validation_size = len(dataset) - train_size
train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

# dataloader to load data into differnet batches
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size = 32, shuffle=False)

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

# training loop
epoch_amount = 10
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

fine_tune_epochs = 10
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


# evaluate the model
model.eval()
validation_loss = 0.0
validation_corrects = 0.0

# save the history of the training model
validation_losses = []
validation_accuracies = []

with torch.no_grad():
    for inputs, labels in validation_dataloader:
        outputs = model(inputs)
        loss = loss_criterion(outputs, labels)

        _, predictions = torch.max(outputs, 1)
        validation_loss = validation_loss + loss.item() * inputs.size(0)
        validation_corrects = validation_corrects + torch.sum(predictions == labels.data)

        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_corrects)
    
validation_loss = validation_loss / validation_size
validation_accuracy = validation_corrects.double() / validation_size

print(f'Validation loss: {validation_loss:.4f}, Validation Accuracy: {validation_accuracy:.4f}')


# plot accuracy of model training in plot
plt.plot(train_losses, 'r', label='train_loss')
plt.plot(validation_losses, 'm', label='validation_loss')
plt.plot(train_accuracies, 'b', label='train_accuracy')
plt.plot(validation_accuracies, 'g', label='validation accuracy')
plt.suptitle('Loss and Accuracy', fontsize=22)
plt.legend(loc='right')
plt.ylabel('Value')
plt.ylim(0, 1)
plt.show()