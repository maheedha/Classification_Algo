import numpy as np
import math
import copy
from sklearn.model_selection import KFold
        
# calculating the mean
def mean(data_set):
    np_arr = np.array(data_set).astype(np.float64)
    return np.mean(np_arr, axis=0)

# calculating the standard deviation   
def std_dev(data_set):
    np_arr = np.array(data_set).astype(np.float64)
    return np.std(np_arr, axis=0)

# calculating the normal distribution
def norm_dist(mean, std, value):
    den = math.sqrt(2*math.pi)
    den = den * std
    num = math.exp((-(value-mean)**2)/(2*std*std))
    pbty = num/den
    return pbty

# calculating tp, tn, fp, fn
def calcMetrics(predictions, test_set):
    test_result = [row[-1] for row in test_set]
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(0, len(predictions)):
        if predictions[i] == test_result[i]:
            if predictions[i] == '1':
                tp = tp + 1
            else:
                tn = tn + 1
        else:
            if predictions[i] == '1':
                fp = fp + 1
            else:
                fn = fn + 1
        
    return tp, tn, fp, fn


def naive_bayes(training_set, test_set):   
# separating data by class
    
    cat_index_list,cat_list = list(), list()
    train_set = copy.deepcopy(training_set)
    for i in range(0, len(train_set[0])):
        if any(c.isalpha() for c in train_set[0][i]):
            cat_index_list.append(i)
    
    for i in cat_index_list:
        cat_list.append([row[i] for row in train_set])
    for row in train_set:
        for i in cat_index_list:
            row[i] = str(0) 
            
    class_map = {}
    col_len = len(train_set[0])
    for i in range(0, len(train_set)):
        try:
            class_map[training_set[i][col_len-1]].append(train_set[i])
        except KeyError:
            class_map[training_set[i][col_len-1]] = [train_set[i]]

    
    mean_map = {}
    std_map = {}
    pbty_map = {}
    key_list = list(class_map.keys())
    a = [row[-1] for row in train_set]
    class_value_count  = dict((i, a.count(i)) for i in a)
    
    cat_pbty = list()
    for i in range(0, len(cat_index_list)):
        d_r = cat_list[i]
        cl_one_map = {}
        cl_zero_map = {}
        for j in range(0, len(d_r)):
            if a[j] == '1':
                try:
                    cl_one_map[d_r[j]] = cl_one_map[d_r[j]] + 1
                except KeyError:
                    cl_one_map[d_r[j]] = 1
            else:
                try:
                    cl_zero_map[d_r[j]] = cl_zero_map[d_r[j]] + 1
                except KeyError:
                    cl_zero_map[d_r[j]] = 1
        cat_pbty.append([cl_zero_map, cl_one_map])
        
    for i in range(0, len(key_list)):
        mean_map[key_list[i]] = mean(class_map[key_list[i]])
        std_map[key_list[i]] = std_dev(class_map[key_list[i]])
    
    # calculating class probabilities
    for i in range(0, len(test_set)):
        for j in range(0, len(key_list)):
            pbty = 1
            for k in range(0, len(test_set[0])-1):
                if k in cat_index_list:
                    d_s = cat_pbty[cat_index_list.index(k)]
                    pbty = pbty * (d_s[int(key_list[j])][test_set[i][k]]/class_value_count[key_list[j]])
                else:
                    pbty = pbty * norm_dist(mean_map[key_list[j]][k], std_map[key_list[j]][k], float(test_set[i][k]))
                    pbty = float(pbty)
            try:
                pbty_map[key_list[j]].append(pbty)
            except KeyError:
                pbty_map[key_list[j]] = [pbty]

    predictions = []  
    for i in range(0, len(test_set)):
        pred = -1
        result = ''
        for j in range(0, len(key_list)):
            if (pbty_map[key_list[j]][i] * (class_value_count[key_list[j]]/len(training_set))) > pred:
                pred = pbty_map[key_list[j]][i] * (class_value_count[key_list[j]]/len(training_set))
                result = key_list[j]
        predictions.append(result)
    return calcMetrics(predictions, test_set)


def read_file(fileName):
    results = []
    with open(fileName) as inputfile:
        for line in inputfile:
            each_line = line.strip().split('\t')
            line_data = []
            for i in range(0, len(each_line)):
                line_data.append(each_line[i])    
            results.append(line_data)
    return results

def divide(partitions, results, partition_len, k):
    dataset_len = 0
    i = 0
    while dataset_len < len(results) and i < k:
        if len(partitions[i]) < partition_len:
                partitions[i].append(results[dataset_len])
                dataset_len = dataset_len + 1
        else:
            i = i + 1
    return partitions

def split(bak_partitions, k, i):
    test_set = bak_partitions[i]
    bak_partitions = np.delete(bak_partitions, (i), axis=0)
    training_set = list()
    for j in range(0, len(bak_partitions)):
        for k in range(0, len(bak_partitions[j])):
            training_set.append(bak_partitions[j][k])
    return training_set, test_set

def kfold(data_set, k, algo):
    partition_len = math.ceil(len(data_set)/k)
    partitions = list()
    for i in range(k):
        partitions.append([])
    partitions = divide(partitions, results, partition_len, k)
    i = k-1
    accuracy_list = list()
    f1_list = list()
    precision_list = list()
    recall_list = list()
    kf = KFold(n_splits=k) # Define the split - into k folds 
    kf.get_n_splits(data_set) # returns the number of splitting iterations in the cross-validator
    KFold(n_splits=k, random_state=None, shuffle=False)
    
    tp, tn, fp, fn = 1, 1, 1, 1
    counter = 0
    for train_index, test_index in kf.split(data_set):
        counter = counter + 1
        training_set, test_set = list(), list()
        for i in train_index:
            training_set.append(data_set[i])
        for i in test_index:
            test_set.append(data_set[i])
        if algo == 1:
            tp, tn, fp, fn = naive_bayes(training_set, test_set)
        if algo == 2:
            tp, tn, fp, fn = knn(training_set, test_set)
        accuracy = (tp+tn)/(tp+tn+fp+fn)
        print('accuracy for iteration ', counter , ': ' , accuracy*100)
        accuracy_list.append(accuracy)
        precision = (tp) / (tp+fp)
        print('precision for iteration ', counter , ': ' , precision*100)
        precision_list.append(precision)
        recall = (tp) / (tp+fn)
        print('recall for iteration ', counter , ': ' , recall*100)
        recall_list.append(recall)
        f_measure = (2 * tp)/ ((2 * tp) + fn + fp)
        print('F-1 Measure for iteration ', counter , ': ' , f_measure*100)
        f1_list.append(f_measure)
    accuracy = (sum(accuracy_list)/float(len(accuracy_list))) * 100
    print('accuracy: ' , accuracy)
    precision = sum(precision_list)/float(len(precision_list))
    print('precision: ', precision)
    recall = sum(recall_list)/float(len(recall_list))
    print('recall: ', recall)
    f_measure = sum(f1_list)/float(len(f1_list))
    print('F-1 Measure: ', f_measure)

results = read_file('dataset2.txt')
kfold(results, 10, 1)




