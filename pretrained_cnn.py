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
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

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

# training loop
epoch_amount = 20
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


