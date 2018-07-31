# Clear any created variables 
#%reset -f

import sys
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def main(dataset_processed_path):
    test_size = 0.2
    random_state = 0
    output_range = range(0,6)
    dataset = pd.read_csv(dataset_processed_path)
    dataset = dataset.dropna()
    
    #Features & Output Split
    X = dataset.iloc[:, 0: 10].values
    y = dataset.iloc[:, 10: 16].values
        
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
    
    #Normalization
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
        
    
    for i in output_range:
        index_of_y_to_classify = i
        if index_of_y_to_classify == 0:
            folder_name = '1-is_normal/'
        elif index_of_y_to_classify == 1:
            folder_name = '2-affected_component/'
        elif index_of_y_to_classify == 2:
            folder_name = '3-scenario/'
        elif index_of_y_to_classify == 3:
            folder_name = '4-operational-scenario/'
        elif index_of_y_to_classify == 4:
            folder_name = '5-combined_affected_component/'
        elif index_of_y_to_classify == 5:
            folder_name = '6-combined_scenario/'
        
        # Begin Classification
        y_current_train = y_train[:, index_of_y_to_classify]
        y_current_test =  y_test[:, index_of_y_to_classify]
                
        # 1- linear regression
        linear_classifier = LogisticRegression(random_state = random_state)
        linear_classifier.fit(X_train, y_current_train)
        cm_linear = pd.crosstab(y_current_test, linear_classifier.predict(X_test))
        accuracies_linear = cross_val_score(estimator=linear_classifier, X = X_train, y = y_current_train, cv = 10)
        
        # 2- KNN
        knn_classifier = KNeighborsClassifier()
        knn_classifier.fit(X_train, y_current_train)
        cm_knn = pd.crosstab(y_current_test, knn_classifier.predict(X_test))
        accuracies_KNN = cross_val_score(estimator=knn_classifier, X = X_train, y = y_current_train, cv = 10)
        
        # 3- SVM
        svm_classifier = SVC(kernel = 'linear', random_state = random_state)
        svm_classifier.fit(X_train, y_current_train)
        cm_svm = pd.crosstab(y_current_test, svm_classifier.predict(X_test))
        accuracies_svm = cross_val_score(estimator=svm_classifier, X = X_train, y = y_current_train, cv = 10)
        
        #4- Kernel SVM
        kernel_svm_classifier = SVC(kernel = 'rbf', random_state = random_state)
        kernel_svm_classifier.fit(X_train, y_current_train)
        cm_kernel_svm = pd.crosstab(y_current_test, kernel_svm_classifier.predict(X_test))
        accuracies_kernel_svm = cross_val_score(estimator=kernel_svm_classifier, X = X_train, y = y_current_train, cv = 10)
        
        #5- Naive Bayes
        naive_classifier = GaussianNB()
        naive_classifier.fit(X_train, y_current_train)
        cm_naive = pd.crosstab(y_current_test, naive_classifier.predict(X_test))
        accuracies_naive = cross_val_score(estimator=naive_classifier, X = X_train, y = y_current_train, cv = 10)
        
        #6- Decision Tree
        decision_tree_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = random_state)
        decision_tree_classifier.fit(X_train, y_current_train)
        cm_decision_tree = pd.crosstab(y_current_test, decision_tree_classifier.predict(X_test))
        accuracies_decision_tree = cross_val_score(estimator=decision_tree_classifier, X = X_train, y = y_current_train, cv = 10)
        
        #7- Random Forest
        random_forest_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = random_state)
        random_forest_classifier.fit(X_train, y_current_train)
        cm_random_forest = pd.crosstab(y_current_test, random_forest_classifier.predict(X_test))
        accuracies_random_forest = cross_val_score(estimator=random_forest_classifier, X = X_train, y = y_current_train, cv = 10)
        
        if os.path.isdir(folder_name) == False:
            os.mkdir(folder_name)
        
        cm_linear.to_csv(folder_name + 'cm_linear.csv')
        cm_knn.to_csv(folder_name + 'cm_knn.csv')
        cm_svm.to_csv(folder_name + 'cm_svm.csv')
        cm_kernel_svm.to_csv(folder_name + 'cm_kernel_svm.csv')
        cm_naive.to_csv(folder_name + 'cm_naive.csv')
        cm_decision_tree.to_csv(folder_name + 'cm_decision_tree.csv')
        cm_random_forest.to_csv(folder_name + 'cm_random_forest.csv')
        
        with open(folder_name + 'CrossValidation.csv', 'a') as cv_file:
            cv_file.write('\nLinear   , ' + str(accuracies_linear.mean()) + " ,   " + str(accuracies_linear.std()))
            cv_file.write('\nNaive    , ' + str(accuracies_naive.mean()) + "  , " + str(accuracies_naive.std()))
            cv_file.write('\nKNN      , ' + str(accuracies_KNN.mean()) + "  , " + str(accuracies_KNN.std()))
            cv_file.write('\nSVM      , ' + str(accuracies_svm.mean()) + "  , " + str(accuracies_svm.std()))
            cv_file.write('\nKern SVM , ' + str(accuracies_kernel_svm.mean()) + " ,  " + str(accuracies_kernel_svm.std()))
            cv_file.write('\nDecision Trees  , ' + str(accuracies_decision_tree.mean()) + ",   " + str(accuracies_decision_tree.std()))
            cv_file.write('\nRandom Forest , ' + str(accuracies_random_forest.mean()) + " ,  " + str(accuracies_random_forest.std()))
            
        
if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = os.getcwd() + '\dataset_processed.csv'
        
    main(path)