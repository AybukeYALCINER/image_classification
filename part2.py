from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
import os
import copy
import tensorflow as tf

# plot the train and validation loss/accuracy plots and save it.
# takes array of validation losses, validation accuracy, train losses, train accuracy and number of epochs respectively.
def plot_graph(val_loss, val_acc, tr_loss, tr_acc, num_epochs):
	plt.subplot(211)
	plt.title("Loss plots vs. Number of Training Epochs")
	plt.plot(range(1,num_epochs+1),val_loss,label="validation")
	plt.plot(range(1,num_epochs+1),tr_loss,label="train")

	plt.xticks(np.arange(1, num_epochs+1, 1.0))
	plt.legend()

	plt.subplot(212)
	plt.title("Accuracy plots vs. Number of Training Epochs")
	plt.plot(range(1,num_epochs+1),val_acc,label="validation")
	plt.plot(range(1,num_epochs+1),tr_acc,label="train")

	plt.xticks(np.arange(1, num_epochs+1, 1.0))
	plt.legend()

	plt.tight_layout()
	plt.savefig("plot.png")

# train the model 
# takes the model, dataloaders, criterion, optimizer, device(GPU or CPU) and number of epochs respectively as parameters
# returns model, array of validation accuracy, validation loss, train accuracy, train loss respectively
def train_model(model, dataloaders, criterion, optimizer, device,num_epochs=25):
    since = time.time()

    val_acc_history = []
    val_loss_history = []
    tr_acc_history = []
    tr_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    
    n_epochs_stop = 5
    min_val_loss = np.Inf
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
		#scheduler.step(epoch) #for lr_scheduler
        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            loader = dataloaders[phase]
            # Iterate over data.
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
#                 print("in dataloaders", end=" ")
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
                    outputs = model(inputs)
#                     print("x")

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'validation':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
            if phase == 'train':
                tr_acc_history.append(epoch_acc)
                tr_loss_history.append(epoch_loss)
			#early stopping
            if phase == 'validation':
              if epoch_loss < min_val_loss:
                epochs_no_improve = 0
                min_val_loss = val_loss

              else:
                epochs_no_improve += 1
                # Check early stopping condition
                if epochs_no_improve == n_epochs_stop:
                  print('Early stopping!')
                  return model, val_acc_history, val_loss_history, tr_acc_history, tr_loss_history

                     

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, val_loss_history, tr_acc_history, tr_loss_history



# data augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validation': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = "dataset"
num_classes = 10
batch_size = 32
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'validation', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=2)
              for x in ['train', 'validation', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation', 'test']}
class_names = image_datasets['train'].classes

model_ft = models.vgg16(pretrained=True)

# freeze layers before classifiers
for param in model_ft.features.parameters():
#     print(param)
    param.requires_grad = False
#different number of layer freeze
#model_ft.features[-1].requires_grad = True
#model_ft.features[-2].requires_grad = True
#model_ft.features[-3].requires_grad = True


model_ft.classifier[6] = nn.Linear(4096,10) #modify the last layer


# specify loss function
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = torch.optim.Adam(model_ft.parameters(), lr=0.001)

#lr_scheduler
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

#different optimizer
#optimizer = torch.optim.SGD(model_ft.parameters(), lr=0.001)

#weight_decay
#optimizer = torch.optim.Adam(model_ft.parameters(), lr=0.1,weight_decay= 0.001)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device) #send the model to the gpu
model_ft, val_acc, val_loss, tr_acc, tr_loss = train_model(model_ft, dataloaders, criterion, optimizer,device, num_epochs=30) #train model

#test the model
correct = 0
topk = 0
total = 0
testloader = dataloaders['test']
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = model_ft(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        probs, classes = outputs.topk(5, dim=1)
        labels_size = labels.size(0)
        for i in range(labels_size):
          if(labels[i] in classes[i]):
            topk += 1
          
        
    print('Accuracy of the model on the test images: %d %%' % (100 * correct / total))
    print('Accuracy of the top 5 on the test images: %d %%' % (100 * topk / total))
	
# val/train loss and accuracy plots
plot_graph(val_loss, val_acc, tr_loss, tr_acc, 30)         





























