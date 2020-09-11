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
from sklearn.metrics import precision_recall_fscore_support as score, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold


def main(dataset_processed_path):
    random_state = 0
    output_range = range(0,6)
    dataset = pd.read_csv(dataset_processed_path)
    dataset = dataset.dropna()
    #Features & Output Split
    X = dataset.iloc[:, 0: 10].values
    y = dataset.iloc[:, 10: 16].values
    
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
            
        
        kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
        
    
        counter = 0
        for train, test in kfold.split(X, y[:, index_of_y_to_classify]):
            X_train = X[train]
            y_train = y[train, index_of_y_to_classify]
            X_test = X[test]
            y_test = y[test, index_of_y_to_classify]
     
        
            #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
                
            #Normalization
            from sklearn.preprocessing import StandardScaler
            sc_X = StandardScaler()
            X_train = sc_X.fit_transform(X_train)
            X_test = sc_X.transform(X_test)            
        
            # Begin Classification
            y_current_train = y_train
            y_current_test =  y_test
            
            # 1- linear regression
            linear_classifier = LogisticRegression(random_state = random_state)
            linear_classifier.fit(X_train, y_current_train)
            cm_linear = pd.crosstab(y_current_test, linear_classifier.predict(X_test))
            accuracy_linear = pd.DataFrame(classification_report(y_current_test, linear_classifier.predict(X_test), output_dict = True)).transpose() 
            #            accuracies_linear = cross_val_score(estimator=linear_classifier, X = X_new, y = y_new, cv = k_cross_val)
            l1, l2, l3, l4 = score(y_current_test, linear_classifier.predict(X_test))
    
            # 2- KNN
            knn_classifier = KNeighborsClassifier()
            knn_classifier.fit(X_train, y_current_train)
            cm_knn = pd.crosstab(y_current_test, knn_classifier.predict(X_test))
            accuracy_knn = pd.DataFrame(classification_report(y_current_test, knn_classifier.predict(X_test), output_dict = True)).transpose() 
#            accuracies_KNN = cross_val_score(estimator=knn_classifier, X = X_new, y = y_new, cv = k_cross_val)
            k1, k2, k3, k4 = score(y_current_test, knn_classifier.predict(X_test))
    
            # 3- SVM
            svm_classifier = SVC(kernel = 'linear', random_state = random_state)
            svm_classifier.fit(X_train, y_current_train)
            cm_svm = pd.crosstab(y_current_test, svm_classifier.predict(X_test))
            accuracy_svm = pd.DataFrame(classification_report(y_current_test, svm_classifier.predict(X_test), output_dict = True)).transpose() 
#            accuracies_svm = cross_val_score(estimator=svm_classifier, X = X_new, y = y_new, cv = k_cross_val)
            s1, s2, s3, s4 = score(y_current_test, svm_classifier.predict(X_test))
    
            #4- Kernel SVM
            kernel_svm_classifier = SVC(kernel = 'rbf', random_state = random_state)
            kernel_svm_classifier.fit(X_train, y_current_train)
            cm_kernel_svm = pd.crosstab(y_current_test, kernel_svm_classifier.predict(X_test))
            accuracy_kernel_svm = pd.DataFrame(classification_report(y_current_test, kernel_svm_classifier.predict(X_test), output_dict = True)).transpose() 
#            accuracies_kernel_svm = cross_val_score(estimator=kernel_svm_classifier, X = X_new, y = y_new, cv = k_cross_val)
            sv1, sv2, sv3, sv4 = score(y_current_test, kernel_svm_classifier.predict(X_test))
    
            #5- Naive Bayes
            naive_classifier = GaussianNB()
            naive_classifier.fit(X_train, y_current_train)
            cm_naive = pd.crosstab(y_current_test, naive_classifier.predict(X_test))
            accuracy_naive =pd.DataFrame(classification_report(y_current_test, naive_classifier.predict(X_test), output_dict = True)).transpose() 
#            accuracies_naive = cross_val_score(estimator=naive_classifier, X = X_new, y = y_new, cv = k_cross_val)
            n1, n2, n3, n4 = score(y_current_test, naive_classifier.predict(X_test))
    
            #6- Decision Tree
            decision_tree_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = random_state)
            decision_tree_classifier.fit(X_train, y_current_train)
            cm_decision_tree = pd.crosstab(y_current_test, decision_tree_classifier.predict(X_test))
            accuracy_decision_tree = pd.DataFrame(classification_report(y_current_test, decision_tree_classifier.predict(X_test), output_dict = True)).transpose() 
#            accuracies_decision_tree = cross_val_score(estimator=decision_tree_classifier, X = X_new, y = y_new, cv = k_cross_val)
            d1, d2, d3, d4 = score(y_current_test, decision_tree_classifier.predict(X_test))
    
            #7- Random Forest
            random_forest_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = random_state)
            random_forest_classifier.fit(X_train, y_current_train)
            cm_random_forest = pd.crosstab(y_current_test, random_forest_classifier.predict(X_test))
            accuracy_random_forest = pd.DataFrame(classification_report(y_current_test, random_forest_classifier.predict(X_test), output_dict = True)).transpose() 
#            accuracies_random_forest = cross_val_score(estimator=random_forest_classifier, X = X_new, y = y_new, cv = k_cross_val)
            r1, r2, r3, r4 = score(y_current_test, random_forest_classifier.predict(X_test))
    
            if os.path.isdir(folder_name) == False:
                os.mkdir(folder_name)
            
            cm_linear.to_csv(folder_name + 'cm_linear.csv')
            cm_knn.to_csv(folder_name + 'cm_knn.csv')
            cm_svm.to_csv(folder_name + 'cm_svm.csv')
            cm_kernel_svm.to_csv(folder_name + 'cm_kernel_svm.csv')
            cm_naive.to_csv(folder_name + 'cm_naive.csv')
            cm_decision_tree.to_csv(folder_name + 'cm_decision_tree.csv')
            cm_random_forest.to_csv(folder_name + 'cm_random_forest.csv')
            
            
            with open(folder_name + 'Accuracies.csv', 'a') as cv_file:
                cv_file.write('\n{}\n'.format(counter))
                cv_file.write('\n{}\n'.format('Linear'))
                accuracy_linear.to_csv(cv_file, header=True)
                cv_file.write('\n{}\n'.format('Naive'))
                accuracy_naive.to_csv(cv_file, header=False)
                cv_file.write('\n{}\n'.format('KNN'))
                accuracy_knn.to_csv(cv_file, header=False)
                cv_file.write('\n{}\n'.format('SVM'))
                accuracy_svm.to_csv(cv_file, header=False)
                cv_file.write('\n{}\n'.format('Kern SVM'))
                accuracy_kernel_svm.to_csv(cv_file, header=False)
                cv_file.write('\n{}\n'.format('DT'))
                accuracy_decision_tree.to_csv(cv_file, header=False)
                cv_file.write('\n{}\n'.format('RF'))
                accuracy_random_forest.to_csv(cv_file, header=False)
                
#                df.to_csv(f, header=False)
#                cv_file.write('\nLinear   , ' + str(accuracy_linear))
#                cv_file.write('\nNaive    , ' + str(accuracy_naive))
#                cv_file.write('\nKNN      , ' + str(accuracy_knn))
#                cv_file.write('\nSVM      , ' + str(accuracy_svm))
#                cv_file.write('\nKern SVM , ' + str(accuracy_kernel_svm))
#                cv_file.write('\nDecision Trees  , ' + str(accuracy_decision_tree))
#                cv_file.write('\nRandom Forest , ' + str(accuracy_random_forest))
#    	
            
#            with open(folder_name + 'CrossValidation.csv', 'a') as cv_file:
#                cv_file.write('\nLinear   , ' + str(accuracies_linear.mean()) + " ,   " + str(accuracies_linear.std()))
#                cv_file.write('\nNaive    , ' + str(accuracies_naive.mean()) + "  , " + str(accuracies_naive.std()))
#                cv_file.write('\nKNN      , ' + str(accuracies_KNN.mean()) + "  , " + str(accuracies_KNN.std()))
#                cv_file.write('\nSVM      , ' + str(accuracies_svm.mean()) + "  , " + str(accuracies_svm.std()))
#                cv_file.write('\nKern SVM , ' + str(accuracies_kernel_svm.mean()) + " ,  " + str(accuracies_kernel_svm.std()))
#                cv_file.write('\nDecision Trees  , ' + str(accuracies_decision_tree.mean()) + ",   " + str(accuracies_decision_tree.std()))
#                cv_file.write('\nRandom Forest , ' + str(accuracies_random_forest.mean()) + " ,  " + str(accuracies_random_forest.std()))
#    	
            with open(folder_name + 'all_scores.csv', 'a') as cv_file:
                cv_file.write('\n{}\n'.format(counter))
                cv_file.write('\nLinear   , ' + 'percision {} \nrecall {}\n {}\nSupport {}'.format(l1, l2, l3, l4))
                cv_file.write('\nNaive    , ' +  'percision {} \nrecall {}\n {}\nSupport {}'.format(n1, n2, n3, n4))
                cv_file.write('\nKNN      , ' +  'percision {} \nrecall {}\n {}\nSupport {}'.format(k1, k2, k3, k4))
                cv_file.write('\nSVM      , ' +  'percision {} \nrecall {}\n {}\nSupport {}'.format(s1, s2, s3, s4))
                cv_file.write('\nKern SVM , ' +  'percision {} \nrecall {}\n {}\nSupport {}'.format(sv1, sv2, sv3, sv4))
                cv_file.write('\nDecision Trees  , ' + 'percision {} \nrecall {}\n {}\nSupport {}'.format(d1, d2, d3, d4))
                cv_file.write('\nRandom Forest , ' +  'percision {} \nrecall {}\n {}\nSupport {}'.format(r1, r2, r3, r4))
            
            counter += 1
            
        
if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = os.getcwd() + '/dataset_processed.csv'
        
    main(path)
