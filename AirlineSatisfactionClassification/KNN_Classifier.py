# NOTE: full_dataset_preprocessed.csv still contains rows with missing values for Arrival Delay in Minutes
# SOURCE: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

# Ignores future warnings
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import itertools

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from torchmetrics.classification import (BinaryAccuracy, BinaryF1Score,
                                         BinaryPrecision, BinaryRecall)

# Split data into 70% training, 10% holdout, and 20% testing
def split_train_test(df):
    seed = 12121212
    train, holdout, test = np.split(df.sample(frac=1, random_state=seed), [int(.7*len(df)), int(.8*len(df))])
    return train, holdout, test

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
def plot_confusion_matrix(pred, label, title):
    plt.clf()
    tn, fp, fn, tp = confusion_matrix(pred, label).ravel()

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

# Split data into train, holdout, and test sets
train, holdout, test = split_train_test(df)

# Delete full df and unneeded holdout set
del df, holdout

# Seperate input features from labels and delete original dataframes
train_X, train_Y = split_X_Y(train)
test_X, test_Y = split_X_Y(test)
del train, test

# Retype and reshape Y sets into numpy array with shape (len(Y), 1)
train_Y = np.array(train_Y).reshape(len(train_Y),1)
test_Y = np.array(test_Y).reshape(len(test_Y),1)

# Initialize model
model = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')

# Train model
model.fit(train_X, train_Y)

# Test model 
test_preds = model.predict(test_X)

# Plot confusion matrix of predictions
plot_confusion_matrix(test_preds.reshape(len(test_preds),1), test_Y, 'KNN 100,000 Confusion Matrix')

# Convert results to tensor for metrics calculations
preds = torch.tensor(test_preds.reshape(len(test_preds),1))
y = torch.tensor(test_Y)

# Instantiate metrics methods
BinAcc = BinaryAccuracy()
BinF1 = BinaryF1Score()
BinRec = BinaryRecall()
BinPrec = BinaryPrecision()

# Print metrics to console
print('\n======== Test Metrics ========')
print(f'Accuracy: {BinAcc(preds, y): .4f} \
        F1 Score: {BinF1(preds, y): .4f} \
        Recall: {BinRec(preds, y): .4f} \
        Precision: {BinPrec(preds, y): .4f}')
