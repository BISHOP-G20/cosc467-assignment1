# NOTE: full_dataset_preprocessed.csv still contains rows with missing values for Arrival Delay in Minutes

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot
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

def train_model(model, train_X, train_Y, val_X, val_Y, criterion, optimizer, epochs):

    for i in range(epochs):
        train_preds = []
        total_train_loss = 0

        print(f'============= Epoch {i + 1} =============')
        for j in tqdm(range(len(train_X))):
            # Reset optimizer
            optimizer.zero_grad()

            # Convert dataframe value to useable tensor
            X = torch.tensor(train_X.iloc[j].astype('float32'))
            Y = torch.tensor([train_Y.iloc[j].astype('float32')])

            # Forward pass
            pred = model(X)
            loss = criterion(pred, Y)
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
                vX = torch.tensor(val_X.iloc[k].astype('float32'))
                vY = torch.tensor([val_Y.iloc[k].astype('float32')])
                val_preds = model(vX)
                val_total_loss += criterion(val_preds, vY).item()

        print(f'Acc: {BinaryAccuracy(train_preds, test_Y)} \
              | Loss: {total_train_loss/len(train_X)} \
              | Val Acc: {BinaryAccuracy(val_preds)} \
              | Val Loss: {val_total_loss/len(val_X)}')
        
def test_model(model, test_X, test_Y, criterion):
    with torch.no_grad():
        preds = model(test_X)
        loss = criterion(preds, test_Y)
        return {'acc':BinaryAccuracy(preds,test_Y),'loss':(loss/len(test_X)),'f1':BinaryF1Score(preds,test_Y),
                'prec':BinaryPrecision(preds,test_Y),'rec':BinaryRecall(preds,test_Y)}



# Model summary
#print(summary(model, (1,23)))

# Only using first 10,026 rows to limit computational expense
# Ignores id column
df = pd.read_csv('C:/Users/gbish/VsCodeProjects/cosc467-assignment1/AirlineSatisfactionClassification/CSVs/full_dataset_preprocessed.csv', usecols={'Gender','Customer Type','Age','Type of Travel','Flight Distance','Inflight wifi service',
                                                                'Departure/Arrival time convenient','Ease of Online booking','Gate location','Food and drink',
                                                                'Online boarding','Seat comfort','Inflight entertainment','On-board service','Leg room service',
                                                                'Baggage handling','Checkin service','Inflight service','Cleanliness','Departure Delay in Minutes',
                                                                'Arrival Delay in Minutes','satisfaction','Class_Flag_1','Class_Flag_2'}, nrows=100)

# Drops columns containing missing values resulting in 10,000 rows containing non-null values
df.dropna(inplace=True)

# Split data into train, validation, and test sets
train, val, test = split_train_test(df)

# Delete full df
del df

# Seperate input features from labels and delete original dataframes
train_X, train_Y = split_X_Y(train)
val_X, val_Y = split_X_Y(val)
test_X, test_Y = split_X_Y(test)
del train, val, test

# Convert dataframes to numpy arrays
#train_X = torch.tensor(train_X.astype('double').values)
#train_Y = torch.tensor(train_Y.astype('double').values)
#val_X = torch.tensor(val_X.astype('double').values)
#val_Y = torch.tensor(val_Y.astype('double').values)
#test_X = torch.tensor(test_X.astype('double').values)
#test_Y = torch.tensor(test_Y.astype('double').values)

# Initialize model
model = NN_Classifier()

# Hyperparameters
epochs = 10
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

train_model(model, train_X, train_Y, val_X, val_Y, criterion, optimizer, epochs)
quit()
metrics = test_model(model, test_X, test_Y, criterion)
acc = metrics['acc']
f1 = metrics['f1']
loss = metrics['loss']
rec = metrics['rec']
prec = metrics['prec']

print('\n======== Test Metrics ========\n')
print(f'Accuracy: {acc} \
        F1 Score: {f1} \
        Loss: {loss} \
        Recall: {rec} \
        Precision: {prec}')