# Clear any created variables 
#%reset -f

import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold

class ResultData:
    def __init__(self, related_classes, count_of_classes_per_instance, hist_of_instances, acc, acc_no_conf, acc_conf):
        self._related_classess = related_classes
        self._count_of_classes_per_instance = count_of_classes_per_instance
        self._hist_of_instances = hist_of_instances
        self._acc = acc
        self._acc_no_conf = acc_no_conf
        self._acc_conf = acc_conf
    

def calculate_accuracy(classifier, x, y):
    confidence = 0.85
    probabilities = classifier.predict_proba(x)
    
    # Accuracy variables
    cnt_all = 0.
    cnt_suggest_first_two_scenarios = 0.
    cnt_with_confidence = 0.
    
    # Hist of non zero probabilities per class
    counts = np.count_nonzero(probabilities, axis = 1)
    unique_counts = np.unique(counts)
    n =  len(unique_counts)
    hist = [0]*n
    for i in range(n):
        hist[i] = np.count_nonzero(counts == unique_counts[i])
    
    # Confusion Matrix (used to check related misclassified scenarios)
    classes = classifier.classes_
    n2  = len(classes)
    related_classes = np.zeros([n2, n2 + 2]) # (1 for all misclassified count and the other for the misclassified and correctly classified by second probability count)

    # Calculate accuracy and build confusion matrix
    for i in range(len(probabilities)):
        indecies = np.argsort(-probabilities[i])
        
        # Report first scenario
        if classes[indecies[0]] == y[i]:
            cnt_all += 1
        
        # Report first and second probable scenarios 
        if classes[indecies[0]] == y[i] or (probabilities[i, indecies[1]] > 0 and classes[indecies[1]] == y[i]):
            cnt_suggest_first_two_scenarios += 1
        
        # Report second probable scenario only if > confidence 
        if (probabilities[i,indecies[0]] >= confidence and classes[indecies[0]] == y[i]) or (probabilities[i,indecies[0]] < confidence and (classes[indecies[0]] == y[i] or (probabilities[i,indecies[1]] > 0 and classes[indecies[1]] == y[i]))):
            cnt_with_confidence +=1 

        index_of_y = classes.tolist().index(y[i])        
        
        if classes[indecies[0]] != y[i]:
            related_classes[index_of_y, n2] += 1
            related_classes[index_of_y, indecies[0]] += 1
        
        if classes[indecies[0]] != y[i] and probabilities[i, indecies[1]] > 0 and classes[indecies[1]]  == y[i]:
            related_classes[index_of_y, n2 + 1] += 1
                     
    res_data = ResultData(related_classes, unique_counts, hist, cnt_all/len(probabilities), cnt_suggest_first_two_scenarios/len(probabilities), cnt_with_confidence/len(probabilities))
    return res_data
    
def write_data_to_file(data, fileName):
    with open(fileName, 'a') as cv_file:
        cv_file.write('Number of predicted classes per instance\n')
        np.savetxt(cv_file, data._count_of_classes_per_instance, delimiter=',', fmt='%1.3f')
        cv_file.write('Number of instances hist (count of instances in each number of predicted classes per instances)\n')    
        np.savetxt(cv_file, data._hist_of_instances, delimiter=',', fmt='%1.3f')
        cv_file.write('Related Classes\n')    
        np.savetxt(cv_file, data._related_classess, delimiter=',', fmt='%1.3f')
    

def main(data_file_path):
    random_state = 0
    
    dataset = pd.read_csv(data_file_path)
    dataset = dataset.dropna()
    X = dataset.iloc[:, 0: 10].values
    y = dataset.iloc[:, 10: 16].values
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
    
    kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
        
    counter = 0
    for train, test in kfold.split(X, y[:, 2]):
        X_train = X[train]
        y_train = y[train, 2]
        X_test = X[test]
        y_test = y[test, 2]
        
        
        # Normalization
        from sklearn.preprocessing import StandardScaler
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)
    
#        index_of_y_to_classify = 2
        folder_name = '3-scenario/fold_{}/'.format(counter)
            
        # Begin Classification
        y_current_train = y_train
        y_current_test =  y_test
        
        # 1- linear regression
        linear_classifier = LogisticRegression(random_state = random_state)
        linear_classifier.fit(X_train, y_current_train)
        accuracies_linear = calculate_accuracy(linear_classifier, X_test, y_current_test)
        
        # 2- KNN
        knn_classifier = KNeighborsClassifier()
        knn_classifier.fit(X_train, y_current_train)
        accuracies_KNN = calculate_accuracy(knn_classifier, X_test, y_current_test)
        
        # 3- SVM
        svm_classifier = SVC(kernel = 'linear', random_state = random_state, probability=True)
        svm_classifier.fit(X_train, y_current_train)
        accuracies_svm = calculate_accuracy(svm_classifier, X_test, y_current_test)
        
        #4- Kernel SVM
        kernel_svm_classifier = SVC(kernel = 'rbf', random_state = random_state, probability=True)
        kernel_svm_classifier.fit(X_train, y_current_train)
        accuracies_kernel_svm = calculate_accuracy(kernel_svm_classifier, X_test, y_current_test)
        
        #5- Naive Bayes
        naive_classifier = GaussianNB()
        naive_classifier.fit(X_train, y_current_train)
        accuracies_naive = calculate_accuracy(naive_classifier, X_test, y_current_test)
        
        #6- Decision Tree
        decision_tree_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = random_state)
        decision_tree_classifier.fit(X_train, y_current_train)
        accuracies_decision_tree = calculate_accuracy(decision_tree_classifier, X_test, y_current_test)
        
        #7- Random Forest
        random_forest_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = random_state)
        random_forest_classifier.fit(X_train, y_current_train)
        accuracies_random_forest = calculate_accuracy(random_forest_classifier, X_test, y_current_test)
        
        if os.path.isdir(folder_name) == False:
            os.mkdir(folder_name)
        
        with open(folder_name + 'Results.csv', 'a') as cv_file:
            cv_file.write('Algorithm, One Scenario, No Confidence, Confidence')
            cv_file.write('\nLinear   , ' + str(accuracies_linear._acc) + "," + str(accuracies_linear._acc_no_conf) + "," +str(accuracies_linear._acc_conf))
            cv_file.write('\nNaive   , ' + str(accuracies_naive._acc) + "," + str(accuracies_naive._acc_no_conf) + "," +str(accuracies_naive._acc_conf))
            cv_file.write('\nKNN      , ' + str(accuracies_KNN._acc) + "," + str(accuracies_KNN._acc_no_conf) + "," +str(accuracies_KNN._acc_conf))
            cv_file.write('\nSVM      , ' + str(accuracies_svm._acc) + "," + str(accuracies_svm._acc_no_conf) + "," +str(accuracies_svm._acc_conf))
            cv_file.write('\nKern SVM , ' + str(accuracies_kernel_svm._acc) + "," + str(accuracies_kernel_svm._acc_no_conf) + "," +str(accuracies_kernel_svm._acc_conf))
            cv_file.write('\nD Trees  , ' + str(accuracies_decision_tree._acc) + "," + str(accuracies_decision_tree._acc_no_conf) + "," +str(accuracies_decision_tree._acc_conf))
            cv_file.write('\nRand For , ' + str(accuracies_random_forest._acc) + "," + str(accuracies_random_forest._acc_no_conf) + "," +str(accuracies_random_forest._acc_conf))
        
        write_data_to_file(accuracies_linear, folder_name + 'Linear.csv')
        write_data_to_file(accuracies_naive, folder_name + 'Naive.csv')
        write_data_to_file(accuracies_KNN, folder_name + 'KNN.csv')
        write_data_to_file(accuracies_svm, folder_name + 'SVM.csv')
        write_data_to_file(accuracies_kernel_svm, folder_name + 'KernelSVM.csv')
        write_data_to_file(accuracies_decision_tree, folder_name + 'DT.csv')
        write_data_to_file(accuracies_random_forest, folder_name + 'Rand Forest.csv')
        counter += 1            

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = os.getcwd() + '/dataset_processed.csv'
        
    main(path)
