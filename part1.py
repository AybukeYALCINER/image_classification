import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import time
import os
import copy

plt.ion()   # interactive mode

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
def feature_extract(dirName):
  feature = []
  label = []
  for inputs, labels in dataloaders[dirName]:
    outputs = model_ft(inputs)
    feature.extend(outputs.cpu().detach().numpy())
    label.extend(labels)
  label = np.array(label)
  return [feature,label]

# shows the number of batch size images and corresponding labels for each image.
# takes images and names of the images as parameters
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# data augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
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
# loads the images from test and train folders who are in dataset folder
# batch size is 4
data_dir = 'dataset'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
# shows the number of batch size images to check 
imshow(out, title=[class_names[x] for x in classes])
# pretrained model is called
model_ft = models.vgg16(pretrained=True)
model_ft.classifier= nn.Sequential(*list(model_ft.classifier.children())[:-2]) # delete the last two layer
# freeze layers
for param in model_ft.parameters():
    param.requires_grad = False
	
# call feature_extract() function and sets the arrays that are returned
test_features = feature_extract('test') 
test_feature = test_features[0]
test_label = test_features[1]

train_features = feature_extract('train')
train_feature = train_features[0]
train_label = train_features[1]


# one vs rest multiclass linear SVM
clf = OneVsRestClassifier(LinearSVC(random_state=0)) # random_state => randomness (optional)
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
