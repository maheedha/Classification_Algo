# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 05:43:35 2017

@author: MaHi
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 22:13:24 2017

@author: MaHi
"""
import numpy as np
import scipy.spatial.distance as dist
import time
import operator
from sklearn.model_selection import KFold
from scipy import stats


def loadData (file_name):
    #Load the data set    
   
    results = []
    with open(file_name) as inputfile:
        for line in inputfile:
            results.append(line.strip().split('\t'))
    
    # Data preprocesing       
    myArray = np.asarray(results)
    
    #Separate the raw data into dataset and labels
    dataset = myArray[:,:-1]
    
    # convert the numpy array to float
    catCol = []
    for ind in range(0,dataset.shape[1]):
        if(dataset[0,ind].isalpha()):
            catCol.append(ind)

    
    catFeature = dataset[:,catCol]
    idx_IN_columns = [i for i in range(0,dataset.shape[1]) if i not in catCol]
    nonCatFeature = dataset[:,idx_IN_columns]
    ds = nonCatFeature.astype(np.float)
    ds=stats.zscore(ds)
            
    #labels coulmn contains the ground truth
    labels =myArray[:,myArray.shape[1]-1]
    return ds,labels,catFeature

        
    
def model(k,dataset,labels,catFeature) :
    
    kf = KFold(n_splits=10) # Define the split - into 10 folds 
    kf.get_n_splits(dataset) # returns the number of splitting iterations in the cross-validator
    KFold(n_splits=10, random_state=None, shuffle=False)
   
    accuracy_list = list()
    f1_list = list()
    precision_list = list()
    recall_list = list()
    for train_index, test_index in kf.split(dataset):
        
        X_train, X_test = dataset[train_index], dataset[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        if(len(catFeature)>0) :
            catFeature_train,catFeature_test=catFeature[train_index],catFeature[test_index]
        predicted = []
        for i in range(X_test.shape[0]) :
            distances = []
            kNeighbour = []
 
            for j in range(X_train.shape[0]) :
                eDist =  dist.euclidean(X_test[i],X_train[j])
                if(len(catFeature)>0) :
                    if(catFeature_test[i]!=catFeature_train[j]) :
                        eDist = eDist +1
                distances.append((eDist,j))
            distances.sort(key=operator.itemgetter(0))
            for ele in range(k):
                kNeighbour.append((y_train[distances[ele][1]]))
            
            cZero=0
            cOne=0
            for p in range(len(kNeighbour)):
                if kNeighbour[p]=="0":
                    cZero = cZero+1
                else :
                    cOne= cOne+1
            if(cZero>cOne) :
                predicted.append(0)
            else :
                predicted.append(1)
        a,p,r,f = metrics_cal(predicted,y_test)
      
        accuracy_list.append(a)
        precision_list.append(p)
        recall_list.append(r)
        f1_list.append(f)
            
    accuracy = (sum(accuracy_list)/float(len(accuracy_list))) * 100
    precision = sum(precision_list)/float(len(precision_list))
    recall = sum(recall_list)/float(len(recall_list))
    f_measure = sum(f1_list)/float(len(f1_list))
    
    return accuracy,precision ,recall,f_measure

def metrics_cal(predicted,y_test) :
    tp, tn, fp, fn = 0, 0, 0, 0
    y_test = list(y_test)

    for index in range(len(predicted)) :
        if(predicted[index] == 1 and y_test[index] =="1" ) :
            tp=tp+1
        elif(predicted[index] == 0 and y_test[index] =="1" ) :
            fn=fn+1
        elif(predicted[index] == 1 and y_test[index] =="0" ) :
            fp=fp+1
        else :
            tn=tn+1
 
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = (tp) / (tp+fp)
    recall = (tp) / (tp+fn)
    f_measure = (2 * tp)/ ((2 * tp) + fn + fp)
    return accuracy,precision,recall,f_measure
        
   
start_time =  time.time()

#Load Input file
inputfile = "dataset2.txt"

# Set value of K
k=50
print("K-Value", k)

#Load the data to get the dataset and labels
dataset,labels,catFeature = loadData(inputfile)
accuracy,precision ,recall,f_measure= model(k,dataset,labels,catFeature)
print('accuracy: ' , accuracy)
print('precision: ', precision)
print('recall: ', recall)
print('F-1 Measure: ', f_measure)
print("Executuion Time ", time.time()-start_time)