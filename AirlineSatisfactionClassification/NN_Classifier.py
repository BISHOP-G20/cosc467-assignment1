# NOTE: full_dataset_preprocessed.csv still contains rows with missing values for Arrival Delay in Minutes

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchsummary import summary
from tqdm import tqdm


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
        x = torch.sigmoid(x)
        x = self.output(x)
        return x

# Split data into 80% training and 20% testing
def split_train_test(df):
    seed = 12121212
    train, val, test = np.split(df.sample(frac=1, random_state=seed), [int(.7*len(df)), int(.8*len(df))])
    return train, val, test

# Seperate inputs from labels
def split_X_Y(df):
    Y = df['satisfaction']
    X = df.drop('satisfaction', axis=1)
    return X, Y

def train(model, train_X, train_Y, val_X, val_Y, criterion, optimizer, epochs):
    for i in range(epochs):
        total_train_correct = 0
        total_train_loss = 0
        print(f'============= Epoch {i + 1} =============')
        for j in tqdm(range(len(train_X))):
            optimizer.zero_grad()
            pred = model(train_X[j])
            loss = criterion(pred, train_Y[j])
            total_train_acc += (pred[0].round() == train_Y[j])
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                val_preds = model(val_X)
                val_loss = criterion(val_preds, val_Y)
                val_acc = (val_preds.round() == val_Y).float().mean()

        print(f'Acc: {total_train_acc/len(train_X)} \
              | Loss: {total_train_loss} \
              | Val Acc: {val_acc} \
              | Val Loss: {val_loss}')



# Model summary
#print(summary(model, (1,23)))

# Only using first 10,026 rows to limit computational expense
# Ignores id column
df = pd.read_csv('CSVs/full_dataset_preprocessed.csv', usecols={'Gender','Customer Type','Age','Type of Travel','Flight Distance','Inflight wifi service',
                                                                'Departure/Arrival time convenient','Ease of Online booking','Gate location','Food and drink',
                                                                'Online boarding','Seat comfort','Inflight entertainment','On-board service','Leg room service',
                                                                'Baggage handling','Checkin service','Inflight service','Cleanliness','Departure Delay in Minutes',
                                                                'Arrival Delay in Minutes','satisfaction','Class_Flag_1','Class_Flag_2'}, nrows=10026)

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
train_X = np.array(train_X)
train_Y = np.array(train_Y)
val_X = np.array(val_X)
val_Y = np.array(val_Y)
test_X = np.array(test_X)
test_Y = np.array(test_Y)


#model = NN_Classifier()