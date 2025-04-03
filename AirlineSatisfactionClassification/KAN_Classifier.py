# NOTE: full_dataset_preprocessed.csv still contains rows with missing values for Arrival Delay in Minutes

# SOURCE: https://github.com/team-daniel/KAN/blob/master/KAN_classification.ipynb
# SOURCE: https://github.com/KindXiaoming/pykan/blob/master/kan/MultKAN.py
# SOURCE: https://kindxiaoming.github.io/pykan/kan.html#kan.MultKAN.MultKAN.evaluate


# Ignores future warnings
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import itertools

import numpy as np
import pandas as pd
import torch
import kan
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch import nn, optim
from torchmetrics.classification import (BinaryAccuracy, BinaryF1Score,
                                         BinaryPrecision, BinaryRecall)
from torchsummary import summary
from tqdm import tqdm

# Accuracy and loss functions for training metrics calculation
def train_train_acc():

    return BinAcc(sigmoid(model(train_dataset['train_input'])), train_dataset['train_label'])

def train_train_loss():
    return criterion(model(train_dataset['train_input']), train_dataset['train_label']) / steps

def train_test_acc():
    return BinAcc(sigmoid(model(train_dataset['test_input'])), train_dataset['test_label'])

def train_test_loss():
    return criterion(model(train_dataset['test_input']), train_dataset['test_label']) / steps

# Plots training and validation accuracy and loss curves
def plot_training_metrics(train_accs, train_losses, val_accs, val_losses, title):
    plt.clf()
    fig, axs = plt.subplots(2)

    X = np.arange(0,len(train_accs))

    fig.suptitle(title)

    axs[0].plot(X, train_accs)
    axs[0].plot(X, val_accs, color='red')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend(['Training','Validation'])

    axs[1].plot(X, train_losses)
    axs[1].plot(X, val_losses, color='red')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].legend(['Training','Validation'])

    plt.savefig('data_graphs/train_val_plots/' + title.replace(' ', '-').replace(',','') + '.png')

# Creates and saves prediction confusion matrix
def plot_confusion_matrix(X, Y, title):
    plt.clf()
    tn, fp, fn, tp = confusion_matrix(Y, np.round(X)).ravel()

    # [tp, tn]
    # [fp, fn]

    cm = np.array([[tp,tn],[fp,fn]])


    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Positive', 'Negative'])
    plt.yticks(tick_marks, ['True', 'False'])

    thresh = cm.max() * .8
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])): 
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig('data_graphs/test_confusion_matrix/' + title.replace(' ', '-').replace(',','') + '.png')

# Split data into 70% training, 10% validation, and 20% testing
def split_train_test(df):
    seed = 12121212
    train, val, test = np.split(df.sample(frac=1, random_state=seed), [int(.7*len(df)), int(.8*len(df))])
    return train, val, test

# Seperate inputs from labels
def split_X_Y(df):
    Y = df['satisfaction']
    X = df.drop('satisfaction', axis=1)
    return X, Y

# Converts all values in df to float32 type for model input
def convert_to_float(df):
    for col in df:
        df[col] = df[col].astype('float32')
    return df

# Only using first nrows to limit computational expense
#   nrows = 100 -> 100 valid datapoints
#   nrows = 1,001 -> 1,000 valid datapoints
#   nrows = 10,026 -> 10,000 valid datapoints
#   nrows = 100,300 -> 100,000 valid datapoints
# Ignores id column
df = pd.read_csv('CSVs/full_dataset_preprocessed.csv', usecols={'Gender','Customer Type','Age','Type of Travel','Flight Distance','Inflight wifi service',
                                                                'Departure/Arrival time convenient','Ease of Online booking','Gate location','Food and drink',
                                                                'Online boarding','Seat comfort','Inflight entertainment','On-board service','Leg room service',
                                                                'Baggage handling','Checkin service','Inflight service','Cleanliness','Departure Delay in Minutes',
                                                                'Arrival Delay in Minutes','satisfaction','Class_Flag_1','Class_Flag_2'}, nrows=100300)

# Drops columns containing missing values resulting in 10,000 rows containing non-null values
df.dropna(inplace=True)

# Convert dataframe values to float32
df = convert_to_float(df)

# Split data into train, validation, and test sets
train, val, test = split_train_test(df)

# Delete full df
del df

# Seperate input features from labels and delete original dataframes
train_X, train_Y = split_X_Y(train)
val_X, val_Y = split_X_Y(val)
test_X, test_Y = split_X_Y(test)
del train, val, test

# Retype datasets sets into torch.tensors
train_X = torch.tensor(np.array(train_X))
train_Y = torch.tensor(np.array(train_Y).reshape(len(train_Y),1))

val_X = torch.tensor(np.array(val_X))
val_Y = torch.tensor(np.array(val_Y).reshape(len(val_Y),1))

test_X = torch.tensor(np.array(test_X))
test_Y = torch.tensor(np.array(test_Y).reshape(len(test_Y),1))

# Recombine training and validation datasets for model training and testing
train_dataset = {'train_input':train_X, 'train_label':train_Y, 'test_input':val_X, 'test_label':val_Y}

# Remove unneeded datasets
del train_X, train_Y, val_X, val_Y

# Initialize model
model = kan.KAN(width=[23,20,1], auto_save=False)

# Hyperparameters
epochs = 15
steps = 70
criterion = nn.BCEWithLogitsLoss()
BinAcc = BinaryAccuracy()
BinF1 = BinaryF1Score()
BinRec = BinaryRecall()
BinPrec = BinaryPrecision()
sigmoid = nn.Sigmoid()

# Training metrics storage
train_accs = []
train_losses = []
val_accs = []
val_losses = []

# Train model
for i in range(epochs):
    print(f'\nEpoch: {i + 1}')
    temp_results = model.fit(train_dataset, opt='Adam', steps=steps, loss_fn=criterion,
                              metrics=(train_train_acc, train_test_acc, train_train_loss, train_test_loss))

    # Average metrics of each st
    temp_train_acc = np.mean(temp_results['train_train_acc'])
    temp_train_loss = np.mean(temp_results['train_train_loss'])
    temp_val_acc = np.mean(temp_results['train_test_acc'])
    temp_val_loss = np.mean(temp_results['train_test_loss'])

    # Store metrics
    train_accs.append(temp_train_acc)
    train_losses.append(temp_train_loss)
    val_accs.append(temp_val_acc)
    val_losses.append(temp_val_loss)
    
    # Print metrics to console
    print(f'Acc: {temp_train_acc: .4f} \
            | Loss: {temp_train_loss: .4f} \
            | Val Acc: {temp_val_acc: .4f} \
            | Val Loss: {temp_val_loss: .4f}')
            
    
# Plot training metrics
plot_training_metrics(train_accs, train_losses, val_accs, val_losses, 'KAN 100,000 Training Metrics')

# Test model
test_pred_logits = model(test_X)
test_preds = sigmoid(test_pred_logits)

# Plot confusion matrix
plot_confusion_matrix(test_preds.detach().numpy(), test_Y, 'KAN 100,000 Confusion Matrix')

# Print metrics to console
print('\n======== Test Metrics ========\n')
print(f"Accuracy: {BinAcc(test_preds, test_Y): .4f} \
        F1 Score: {BinF1(test_preds, test_Y): .4f} \
        Loss: {criterion(test_pred_logits,test_Y): .4f} \
        Recall: {BinRec(test_preds, test_Y): .4f} \
        Precision: {BinPrec(test_preds, test_Y): .4f}")