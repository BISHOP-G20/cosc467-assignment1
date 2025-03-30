import itertools

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

df = pd.read_csv('CSVs/full_dataset.csv', usecols={'Gender','Customer Type','Age','Type of Travel','Class','Flight Distance','Inflight wifi service',
                                             'Departure/Arrival time convenient','Ease of Online booking','Gate location','Food and drink','Online boarding',
                                             'Seat comfort','Inflight entertainment','On-board service','Leg room service','Baggage handling','Checkin service',
                                             'Inflight service','Cleanliness','Departure Delay in Minutes','Arrival Delay in Minutes','satisfaction'})

def plot_2cat_vs_2cat(col_name1, col_name2, df, cmap=plt.cm.Blues):
    classes = [col_name1, col_name2]
    simple_df = df[[classes[0],classes[1]]]
    class1_values = simple_df[classes[0]].unique()
    class2_values = simple_df[classes[1]].unique()

    # [(0,0), (1,0)]
    # [(0,1), (1,1)]
    
    cm = np.array([[((simple_df[classes[0]] == class1_values[0]) & (simple_df[classes[1]] == class2_values[0])).sum(), # male and dissatisfied
          ((simple_df[classes[0]] == class1_values[1]) & (simple_df[classes[1]] == class2_values[0])).sum()], # female and dissatisfied
          [((simple_df[classes[0]] == class1_values[0]) & (simple_df[classes[1]] == class2_values[1])).sum(), # male and satisfied
           ((simple_df[classes[0]] == class1_values[1]) & (simple_df[classes[1]] == class2_values[1])).sum()]])# female and satisfied

    print('col_name1: ', col_name1)
    print('col_name1: ', col_name2)
    print('col 1 vals: ', class1_values)
    print('col2 vals: ', class2_values)
    print('cm: ', cm)
    print('cm sum: ', sum(cm[0]) + sum(cm[1]))
   
    plt.imshow(cm, cmap=cmap)
    plt.title(col_name1 + ' vs ' + col_name2)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, class1_values)
    plt.yticks(tick_marks, class2_values)

    thresh = cm.max() * .8
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])): 
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.xlabel(classes[0])
    plt.ylabel(classes[1])
    plt.savefig('data_graphs/heat_maps/' + col_name1.lower().replace(' ','-') + '_' + col_name2.lower().replace(' ','-') + '.png')

def plot_scatter(x_col_name, y_col_name, df):
    X = df[x_col_name]
    Y = df[y_col_name]
    plt.scatter(X,Y)
    plt.title(y_col_name + ' vs ' + x_col_name)
    plt.xlabel(x_col_name)
    plt.ylabel(y_col_name)
    plt.savefig('data_graphs/scatter_plots/' + x_col_name.lower().replace(' ','-') + '_' + y_col_name.lower().replace(' ','-') + '.png')
    
def plot_sat_vs_sat_histogram(x_col_name, y_col_name, df):
    X = np.arange(0,6)
    Y = np.zeros(6)

    for i in range(6):
        Y[i] = np.mean(df[df[x_col_name] == i][y_col_name])

    plt.bar(X,Y)
    plt.title(y_col_name + ' vs ' + x_col_name)
    plt.xlabel(x_col_name)
    plt.ylabel(y_col_name)
    plt.savefig('data_graphs/histograms/' + x_col_name.lower().replace(' ','-').replace('/','-') + '_' + y_col_name.lower().replace(' ','-').replace('/','-') + '.png') 

def plot_cat_count_histogram(col_name, df):
    values = df[col_name].value_counts()
    X = df[col_name].unique()
    Y = []

    for i in range(len(X)):
        Y.append(values[X[i]])

    plt.bar(X,Y)
    plt.title(col_name + ' category counts')
    plt.xlabel(col_name)
    plt.ylabel('count')
    plt.savefig('data_graphs/feature_graphs/' + col_name.lower().replace(' ', '-').replace('/','-') + '.png')

def plot_rating_hist(col_name, df):
    values = df[col_name].value_counts()
    X = df[col_name].unique()
    Y = []

    for i in range(len(X)):
        Y.append(values[X[i]])

    plt.bar(X,Y)
    plt.title(col_name + ' rating counts')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.savefig('data_graphs/feature_graphs/' + col_name.lower().replace(' ','-').replace('/','-') + '.png')

def plot_binned_hist(col_name, df):
    plt.hist(df[col_name], bins=5, edgecolor='black')
    plt.title(col_name + ' distribution')
    plt.xlabel(col_name)
    plt.ylabel('Frequency')
    plt.savefig('data_graphs/feature_graphs/' + col_name.lower().replace(' ','-').replace('/','-') + '.png')
    