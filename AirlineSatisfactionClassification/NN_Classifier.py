# NOTE: full_dataset_preprocessed.csv still contains rows with missing values for Arrival Delay in Minutes

# Ignores future warnings
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import itertools

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch import nn, optim
from torchmetrics.classification import (BinaryAccuracy, BinaryF1Score,
                                         BinaryPrecision, BinaryRecall)
from torchsummary import summary
from tqdm import tqdm

use_cuda = torch.cuda.is_available()
print(use_cuda)

class NN_Classifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.input = nn.Linear(23, 100)
        self.relu = nn.ReLU()
        self.hidden = nn.Linear(100, 50)
        self.output = nn.Linear(50, 1) 

    def forward(self, x):
        x = self.input(x)
        x = self.relu(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = torch.sigmoid(x)
        return x

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

# Creates and saves prediction confusion matrix
def plot_confusion_matrix(X, Y, title):
    plt.clf
    tn, fp, fn, tp = confusion_matrix(Y, X).ravel()

    # [tp, tn]
    # [fp, fn]

    cm = np.array([[tp,tn],[fp,fn]])


    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('NN Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Positive', 'Negative'])
    plt.yticks(tick_marks, ['True', 'False'])

    thresh = cm.max() * .8
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])): 
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig('data_graphs/test_confusion_matrix/' + title.replace(' ', '-').replace(',','') + '.png')

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

def train_model(model, train_X, train_Y, val_X, val_Y, criterion, optimizer, epochs, title):

    train_accs = []
    train_losses = []
    val_accs = []
    val_losses = []

    for i in range(epochs):
        train_preds = []
        total_train_loss = 0

        print(f'\n============= Epoch {i + 1} =============')
        for j in tqdm(range(len(train_X))):
            # Reset optimizer
            optimizer.zero_grad()

            # Convert dataframe value to useable tensor
            X = torch.tensor(train_X.iloc[j])
            Y = torch.tensor(train_Y[j])

            # Forward pass
            pred = model(X)
            loss = criterion(pred, Y)
            train_preds.append([pred.item()])
            total_train_loss += loss.item()

            # Backward pass and update
            loss.backward()
            optimizer.step()

        # Storage for validation metrics
        val_preds = []
        val_total_loss = 0

        # Validation pass
        with torch.no_grad():
            for k in range(len(val_X)):

                # Convert dataframe value to useable tensor
                vX = torch.tensor(val_X.iloc[k])
                vY = torch.tensor(val_Y[k])

                # Make predictions
                val_pred = model(vX)

                # Store validation metrics
                val_preds.append([val_pred.item()])
                val_total_loss += criterion(val_pred, vY).item()

        # Calculate training and validation metrics
        temp_train_acc = BinAcc(torch.tensor(train_preds), torch.tensor(train_Y))
        temp_train_loss = total_train_loss/len(train_X)
        temp_val_acc = BinAcc(torch.tensor(val_preds), torch.tensor(val_Y))
        temp_val_loss = val_total_loss/len(val_X)

        # Store metrics from epoch in array for plotting
        train_accs.append(temp_train_acc)
        train_losses.append(temp_train_loss)
        val_accs.append(temp_val_acc)
        val_losses.append(temp_val_loss)
        
        # Print metrics to console
        print(f'Acc: {temp_train_acc: .4f} \
              | Loss: {temp_train_loss: .4f} \
              | Val Acc: {temp_val_acc: .4f} \
              | Val Loss: {temp_val_loss: .4f}')
        
        # Delete temporary values
        del temp_train_acc, temp_train_loss, temp_val_acc, temp_val_loss

    plot_training_metrics(train_accs, train_losses, val_accs, val_losses, title)
        
def test_model(model, test_X, test_Y, criterion, title):
    with torch.no_grad():
        preds = []
        total_loss = 0
        print('\nTESTING')
        for i in tqdm(range(len(test_X))):
            X = torch.tensor(test_X.iloc[i])
            Y = torch.tensor(test_Y[i])

            pred = model(X)
            preds.append([pred.item()])
            total_loss += criterion(pred, Y).item()

        # Plot and save confusion matrix
        plot_confusion_matrix(np.round(np.array(preds)), test_Y, title)

        t_preds = torch.tensor(preds)
        t_Y = torch.tensor(test_Y)
        return {'acc':BinAcc(t_preds, t_Y),'loss':(total_loss/len(test_X)),
                'f1':BinF1(t_preds, t_Y),'prec':BinPrec(t_preds, t_Y),
                'rec':BinRec(t_preds, t_Y)}



# Model summary
#print(summary(model, (1,23)))

# Only using first 10,026 rows to limit computational expense
#   nrows = 100 -> 100 valid datapoints
#   nrows = 10,026 -> 10,000 valid datapoints
# Ignores id column
df = pd.read_csv('CSVs/full_dataset_preprocessed.csv', usecols={'Gender','Customer Type','Age','Type of Travel','Flight Distance','Inflight wifi service',
                                                                'Departure/Arrival time convenient','Ease of Online booking','Gate location','Food and drink',
                                                                'Online boarding','Seat comfort','Inflight entertainment','On-board service','Leg room service',
                                                                'Baggage handling','Checkin service','Inflight service','Cleanliness','Departure Delay in Minutes',
                                                                'Arrival Delay in Minutes','satisfaction','Class_Flag_1','Class_Flag_2'}, nrows=10026)

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

# Retype and reshape Y sets into numpy array with shape (len(Y), 1)
train_Y = np.array(train_Y).reshape(len(train_Y),1)
val_Y = np.array(val_Y).reshape(len(val_Y),1)
test_Y = np.array(test_Y).reshape(len(test_Y),1)

# Initialize model
model = NN_Classifier()

# Hyperparameters
epochs = 15
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())
BinAcc = BinaryAccuracy()
BinF1 = BinaryF1Score()
BinRec = BinaryRecall()
BinPrec = BinaryPrecision()

# Train model
train_model(model, train_X, train_Y, val_X, val_Y, criterion, optimizer, epochs, 'NN 10,000 Training Metrics')

# Test model
metrics = test_model(model, test_X, test_Y, criterion, 'NN 10,000 Confusion Matrix')

# Print test metrics
acc = metrics['acc']
f1 = metrics['f1']
loss = metrics['loss']
rec = metrics['rec']
prec = metrics['prec']

print('\n======== Test Metrics ========\n')
print(f'Accuracy: {acc: .4f} \
        F1 Score: {f1: .4f} \
        Loss: {loss: .4f} \
        Recall: {rec: .4f} \
        Precision: {prec: .4f}')