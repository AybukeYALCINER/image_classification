## Access the dataset: https://drive.google.com/open?id=1XzLpuQ-jtqXgU-SsFKLrNPJOrvtmtyE3

Use python-3 and pytorch library. <br>

To use colab add the dataset file to the drive and write :<br>

from google.colab import drive<br>
drive.mount('/content/drive')<br>

and give the path of the dataset something like "drive/My Drive/dataset"<br>

We have 10 different classes and we have 400 training, 250 validation and 250 test images in each class.<br>

## PART-I ##

In this part, pretrained VGG-16 model is used(on imageNet) and this model is used as feature extractor(from FC7 layer after RELU).
Then the extracted features are given to the one-vs-rest multiclass linear SVM to classify. We just use train and test set. 
This part is run on CPU.<br>

And there are some functions: <br>

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues) => # generate the confusion matrix of the predictions<br>
-- takes correct labels, predicted labels, class names,boolean normalize, title of the graph and background colors of the graph as parameter respectively 
<br>
def feature_extract(dirName) => # returns an array that holds array of extracted features of the images and array of the class names of the images.<br>
  takes dirName as parameter that specify the which images extracted.
<br>
def imshow(inp, title=None) => # shows the number of batch size images and corresponding labels for each image. and  takes images and names of the images as parameters
<br>

## PART-II ##

In this part, we finetune the VGG-16 model of the CNN which is pretrained on imageNet. After we finetune the model, we test it and return the top-1 and top-5 accuracy.<br>
We use train,validation and test sets. This part is run on GPU. To run on CPU, you can remove the "device" and early stopping is made to avoid it, comment out the lines that cause he early stopping.<br>
NOT: We modify the last layer of the model because we have 10 classes and imgaNet has 1000 classes.<br>

And there are some functions: 
<br>
def plot_graph(val_loss, val_acc, tr_loss, tr_acc, num_epochs) => # plot the train and validation loss/accuracy plots and save it.<br>
 takes array of validation losses, validation accuracy, train losses, train accuracy and number of epochs respectively.
<br>
def train_model(model, dataloaders, criterion, optimizer, device,num_epochs=25) => # train the model <br>
 takes the model, dataloaders, criterion, optimizer, device(GPU or CPU) and number of epochs respectively as parameters<br>
 returns model, array of validation accuracy, validation loss, train accuracy, train loss respectively<br>

## PART-III ##

In this part, pretrained VGG-16 model is used(on imageNet) and after finetune the model, used as feature extractor(from FC7 layer after RELU).<br>
Then the extracted features are given to the one-vs-rest multiclass linear SVM to classify. This part is run on GPU. To run on CPU, you can remove the "device"<br>
NOT: We modify the last layer of the model because we have 10 classes and imgaNet has 1000 classes.<br>

And there are some functions: <br>

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues) => # generate the confusion matrix of the predictions<br>
takes correct labels, predicted labels, class names,boolean normalize, title of the graph and background colors of the graph as parameter respectively <br>

def feature_extract(dirName) => # returns an array that holds array of extracted features of the images and array of the class names of the images.<br>
 takes dirName as parameter that specify the which images extracted<br>

def train_model(model, dataloaders, criterion, optimizer, device,num_epochs=25) => # train the model <br>
 takes the model, dataloaders, criterion, optimizer, device(GPU or CPU) and number of epochs respectively as parameters<br>
 returns model, array of validation accuracy, validation loss, train accuracy, train loss respectively<br>
