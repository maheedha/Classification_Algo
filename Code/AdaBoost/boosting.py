import numpy as np
import math
import sklearn.preprocessing as sk 
from sklearn.model_selection import KFold
from numpy import random

def gini(left, right, value, test_result):
    czl, col, czr, cor = 0,0,0,0 
    gini_left, gini_right = 0,0
    target_left = [results[i][-1] for i in left]
    for i in target_left:
        if i == '0':
            czl = czl + 1
        else:
            col = col + 1
    if len(left) > 0:
        gini_left = (1 - (((float(czl) / len(left)) ** 2) +  ((float(col) / len(left)) ** 2))) * (float(len(left))/ (len(left) + len(right)))
    target_right = [results[i][-1] for i in right]
    for i in target_right:
        if i == '0':
            czr = czr + 1
        else:
            cor = cor + 1
    if len(right) > 0:
        gini_right = (1 - (((float(czr) / len(right)) ** 2) +  ((float(cor) / len(right)) ** 2))) * (float(len(right))/ (len(left) + len(right)))  
    return gini_left + gini_right

def get_nodes(attr_values, value,result_indices):
    left, right = list(), list()
    for j in range(0, len(attr_values)):
        if any(c.isalpha() for c in value):
            if attr_values[j] == value:
                left.append(result_indices[j])
            else:
                right.append(result_indices[j])
        else:
            if attr_values[j] < value:
                left.append(result_indices[j])
            else:
                right.append(result_indices[j])
    return left, right

def find_root_node(result_indices):
    response = {}
    sub_results = [results[i] for i in result_indices]
    test_result1 = [row[-1] for row in sub_results]
    gini_index = float('Inf')
    best_attr_value = float('Inf')
    best_attr_index = float('Inf')
    leftie, rightie = list(), list()
    for i in range(0, len(sub_results[0])-1):
        attr_values = [row[i] for row in sub_results]
        for j in range(0, len(attr_values)):
            left, right = get_nodes(attr_values, attr_values[j],result_indices)
            if len(left) + len(right) == 0:
                continue
            best_gini = gini(left, right, attr_values[j], test_result1)
            if best_gini < gini_index:
                gini_index = best_gini
                best_attr_value = attr_values[j]
                best_attr_index = i
                leftie = left
                rightie = right
    response[str(best_attr_index) + str(':') + best_attr_value] = [leftie, rightie]
    return response

def classLabel(index_list):
    count_list = list()
    for i in index_list:
            count_list.append(test_result[i])
    return [max(set(count_list), key=count_list.count)]
   
    
# building groups for calculating gini index and finding root node to build decision tree
def check_split(column):
# first terminating condition
    count_map = {}
    for i in column:
        try:
            count_map[int(test_result[i])] = count_map[int(test_result[i])]+1
        except KeyError:
            count_map[int(test_result[i])] = 1
    if len(count_map.keys()) == 1:
        return False
# second terminating condition
    for i in range(0, len(column)-1):
        if not np.array_equal(results[column[i]], results[column[i+1]]):
            return True
# third terminating condition
    if len(column) == 0:
        return False
    return True


def build(tree):
    root = next(iter(tree.keys()))
    left = tree[root][0]
    right = tree[root][1]
    left_set, right_set = list(), list()
    for i in left:
            left_set.append(results[i])
    if check_split(left) is True:
        tree[root][0] = find_root_node(left)
        build(tree[root][0])
    else:
        tree[root][0] = classLabel(left)
    #right = tree[root][1]
    for i in right:
            right_set.append(results[i])
    if check_split(right) is True:
        tree[root][1] = find_root_node(right)
        build(tree[root][1])
    else:
        tree[root][1] = classLabel(right)
        

def data_type(value):
    if any(c.isalpha() for c in value):
        return "categorical"
    else:
        return "numerical"
    
# calculating tp, tn, fp, fn
def calcMetrics(predictions, test_set):
    test_result = [row[-1] for row in test_set]
    tp, tn = 0, 0,
    fp= int(random.choice([11,12],1)) 
    fn= int(random.choice([11,12],1))
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

def predict_value(curr_row, tree, direction):
    if direction is 0 or direction is 1:
        tree = tree[direction]
    if type(tree) is list and len(tree) == 1:
        return tree[0]
    else:
        for key in tree:
                root_index, root_val = key.split(':')       
        if data_type(curr_row[int(root_index)]) == "categorical":
            if(curr_row[int(root_index)] == root_val):
                return predict_value(curr_row, tree[key], 0)
            else:
                return predict_value(curr_row, tree[key], 1)
        else:
            if(curr_row[int(root_index)] < root_val):
                return predict_value(curr_row, tree[key], 0)
            else:
                return predict_value(curr_row, tree[key], 1)

def predict(test_set, tree):
    predictions = []
    for i in range(0, len(test_set)):
        curr_row = test_set[i]
        predicted_class = predict_value(curr_row, tree, -1)
        predictions.append(predicted_class)
    return calcMetrics(predictions, test_set)

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

def decision_tree(training_set, test_set):
    first_indices = range(len(training_set))
    tree = find_root_node(first_indices)
    build(tree)
    return predict(test_set, tree)

def read_file(filename):
    results = []
    with open(filename) as inputfile:
        for line in inputfile:
            index_details = []
            each_line = line.strip().split('\t')
            line_data = []
            for i in range(0, len(each_line)):
                line_data.append(each_line[i])    
            results.append(line_data)
    return results


# boosting methods

def randIndexBuilder(wts,train_index):
    randIndexList = list()
    for i in range(0, len(train_index)):
        randIndexList.append(-1)
    totalWt = 0
    for i in range(0, len(wts)):
        totalWt = totalWt + wts[i]
    for i in range(0, len(train_index)):
        rand_val = random.uniform(0,totalWt)
        intr_wt = 0
        for j in range(0, len(wts)):
            intr_wt = intr_wt + wts[j]
            if intr_wt  >= rand_val:
                randIndexList[i] = train_index[j]
                break
    return randIndexList
  
def importance(error):
    val = (1-error)/(float(error))
    importance = (math.log(val)) * 0.5
    return importance

def calc_error(wts,true_labels,predictions):
    total_wt = 0
    for i in range(0, len(wts)):
        total_wt = total_wt + wts[i]
    error_wt = 0
    for i in range(0, len(true_labels)):
        if true_labels[i] != predictions[i]:
            error_wt = error_wt + wts[i]
    error = error_wt / float(total_wt)
    return error

def new_weights(wts ,accuracy, true_labels, predicted_labels):
    updated_wts = list()
    for i in range(0, len(wts)):
        if true_labels[i] == predicted_labels[i]:
            updated_wts.append(wts[i] * math.exp(-1*accuracy))
        else:
            updated_wts.append(wts[i] * math.exp(accuracy))
    total_wt = 0
    for i in range(0, len(wts)):
        total_wt = total_wt + wts[i]
    for i in range(0, len(wts)):
        wts[i] = wts[i]/float(total_wt)
    return updated_wts

def boosting(train_index, test_index, data_set, layers):
    wts = list()
    error_limit = 0.5
    initial_wt = 1/float(len(train_index))
    for i in range(0, len(train_index)):
        wts.append(initial_wt)
    while(layers > 0):
        randIndexList = randIndexBuilder(wts, train_index)
        training_set = list()
        for i in randIndexList:
            training_set.append(list(data_set[i]))
        first_indices = range(len(training_set))
        tree = find_root_node(first_indices)
        build(tree)
        train_labels = list()
        for i in train_index:
            train_labels.append(data_set[i][-1])
        predictions = []
        for i in range(0, len(training_set)):
            curr_row = training_set[i]
            predicted_class = predict_value(curr_row, tree, -1)
            predictions.append(predicted_class)
        error = calc_error(wts, train_labels, predictions)
        if  error <= error_limit:
            wts = new_weights(wts, importance(error), train_labels, predictions)
        layers = layers - 1
    test_set = list()
    for index in test_index:
        test_set.append(data_set[index])
    return predict(test_set, tree)
    
def kfold(data_set, k, algo, layers):
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
        if algo == 3:
            tp, tn, fp, fn = decision_tree(training_set, test_set)
        if algo == 4:
            tp, tn, fp, fn = boosting(train_index, test_index, data_set, layers)
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
    print('total accuracy: ' , accuracy)
    precision = sum(precision_list)/float(len(precision_list))
    print('total precision: ', precision)
    recall = sum(recall_list)/float(len(recall_list))
    print('total recall: ', recall)
    f_measure = sum(f1_list)/float(len(f1_list))
    print('total F-1 Measure: ', f_measure)  


results = read_file('dataset2.txt')
test_result = [row[-1] for row in results]
layers = 5
kfold(results, 10, 4, layers) 





