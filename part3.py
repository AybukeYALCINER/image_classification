from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import time
import os
import copy

# generate the confusion matrix of the predictions
# takes correct labels, predicted labels, class names,boolean normalize, title of the graph and background colors of the graph as parameter respectively 
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
   
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig("conf_mtrx_prt1_b4.png")
    return ax
	
# returns an array that holds array of extracted features of the images and array of the class names of the images.
# takes dirName as parameter that specify the which images extracted
def feature_extract(dirName,device):
  feature = []
  label = []
  for inputs, labels in dataloaders[dirName]:
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model_ft(inputs)
    feature.extend(outputs.cpu().numpy())
    label.extend(labels.cpu().numpy())
  return [feature,label]

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
    
   
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

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

#freeze layers
for param in model_ft.features.parameters():
    param.requires_grad = False
model_ft.features[-1].requires_grad = True
model_ft.features[-2].requires_grad = True
model_ft.features[-3].requires_grad = True

#modify last layer
model_ft.classifier[6] = nn.Linear(4096,10)

# specify loss function
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = torch.optim.Adam(model_ft.parameters(), lr=0.001)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)
model_ft, val_acc, val_loss, tr_acc, tr_loss = train_model(model_ft, dataloaders, criterion, optimizer,device, num_epochs=30) #train model

model_ft.classifier= nn.Sequential(*list(model_ft.classifier.children())[:-2]) # delete the last two layer
# freeze all layers
for param in model_ft.parameters():
    param.requires_grad = False
	
#extract feaatures
test_features = feature_extract('test',device) 
test_feature = test_features[0]
test_label = test_features[1]

train_features = feature_extract('train',device)
train_feature = train_features[0]
train_label = train_features[1]


# one vs rest multiclass linear SVM
clf = OneVsRestClassifier(LinearSVC(random_state=0,max_iter = 20000)) # random_state => randomness (optional)
classifier = clf.fit(train_feature, train_label)

# predictions
length_test = len(test_feature)
class_based_acc = np.zeros(10)
for test_ind in range(length_test):
  y_pred = classifier.predict([test_feature[test_ind]])
  if(y_pred == test_label[test_ind]):
    class_based_acc[int(y_pred)] += 1

#print class based and general acc
for acc_ind in range(10):
  print("Accuracy of class "+ str(acc_ind)+ " "+str(class_based_acc[acc_ind]*100/length_test))
acc = classifier.score(test_feature,test_label)
print("Accuracy: "+ str(acc*100))
y_pred = classifier.predict(test_feature)
# confusion matrix
plot_confusion_matrix(test_label, y_pred, classes=class_names, normalize=True, title='Normalized confusion matrix')




























