################################
# author: Aybüke YALÇINER.     #
# number: 21527544             #
# Assignment2                  #
################################

Use python-3 and pytorch library. 

To use colab add the dataset file to the drive and write :

from google.colab import drive
drive.mount('/content/drive')

and give the path of the dataset something like "drive/My Drive/dataset"

We have 10 different classes and we have 400 training, 250 validation and 250 test images in each class.

## PART-I ##

In this part, pretrained VGG-16 model is used(on imageNet) and this model is used as feature extractor(from FC7 layer after RELU).
Then the extracted features are given to the one-vs-rest multiclass linear SVM to classify. We just use train and test set. 
This part is run on CPU.

And there are some functions: 

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues) => # generate the confusion matrix of the predictions
# takes correct labels, predicted labels, class names,boolean normalize, title of the graph and background colors of the graph as parameter respectively 

def feature_extract(dirName) => # returns an array that holds array of extracted features of the images and array of the class names of the images.
# takes dirName as parameter that specify the which images extracted.

def imshow(inp, title=None) => # shows the number of batch size images and corresponding labels for each image.
# takes images and names of the images as parameters

## PART-II ##

In this part, we finetune the VGG-16 model of the CNN which is pretrained on imageNet. After we finetune the model, we test it and return the top-1 and top-5 accuracy.
We use train,validation and test sets. This part is run on GPU. To run on CPU, you can remove the "device" and early stopping is made to avoid it, comment out the lines that cause he early stopping.
NOT: We modify the last layer of the model because we have 10 classes and imgaNet has 1000 classes.

And there are some functions: 

def plot_graph(val_loss, val_acc, tr_loss, tr_acc, num_epochs) => # plot the train and validation loss/accuracy plots and save it.
# takes array of validation losses, validation accuracy, train losses, train accuracy and number of epochs respectively.

def train_model(model, dataloaders, criterion, optimizer, device,num_epochs=25) => # train the model 
# takes the model, dataloaders, criterion, optimizer, device(GPU or CPU) and number of epochs respectively as parameters
# returns model, array of validation accuracy, validation loss, train accuracy, train loss respectively

## PART-III ##

In this part, pretrained VGG-16 model is used(on imageNet) and after finetune the model, used as feature extractor(from FC7 layer after RELU).
Then the extracted features are given to the one-vs-rest multiclass linear SVM to classify. This part is run on GPU. To run on CPU, you can remove the "device"
NOT: We modify the last layer of the model because we have 10 classes and imgaNet has 1000 classes.

And there are some functions: 

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues) => # generate the confusion matrix of the predictions
# takes correct labels, predicted labels, class names,boolean normalize, title of the graph and background colors of the graph as parameter respectively 

def feature_extract(dirName) => # returns an array that holds array of extracted features of the images and array of the class names of the images.
# takes dirName as parameter that specify the which images extracted

def train_model(model, dataloaders, criterion, optimizer, device,num_epochs=25) => # train the model 
# takes the model, dataloaders, criterion, optimizer, device(GPU or CPU) and number of epochs respectively as parameters
# returns model, array of validation accuracy, validation loss, train accuracy, train loss respectively