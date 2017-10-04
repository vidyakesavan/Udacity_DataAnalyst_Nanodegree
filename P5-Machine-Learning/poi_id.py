#!/usr/bin/python

# All the imports

import sys
import pickle
import pandas as pd
import numpy as np
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV


## ----------------------------------------------------------------------------
##                          SELECT THE FEATURES
## ----------------------------------------------------------------------------

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi','deferred_income',
                 'email_address','exercised_stock_options',
                 'expenses','long_term_incentive',
                 'other','restricted_stock','shared_receipt_with_poi',
                 'total_payments','total_stock_value', 'from_poi_to_this_person',
                 'from_this_person_to_poi', 'to_messages', 'from_messages']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

## ----------------------------------------------------------------------------
##                          EXPLORE THE DATA
## ----------------------------------------------------------------------------

    
### How many records are available in the dataset?
print "\nLength of the dataset is: ", len(data_dict)

### Print the number of POIs identified in the dataset
print "\nCount of POIs in the dataset is: ", \
len([v for v in data_dict.values() if v['poi'] == True])

### Print the number of non-POIs in the dataset
print "\nCount of Non-POIs in the dataset is: ", \
len([v for v in data_dict.values() if v['poi'] == False])

### Number of features
keys = next(data_dict.itervalues()).keys()
print "\nNumber of features in the dataset is: ",len(keys)

### Number of NaN's for each feature
NaN_per_feature = {}
for k,v in data_dict.iteritems():
    for k2,v2 in v.iteritems():
        if v2 == 'NaN':
            if k2 in NaN_per_feature:
                NaN_per_feature[k2] += 1
            else:
                NaN_per_feature[k2] = 1
                
import pprint as pp
print "\nNumber of NaN's per feature is: "
pp.pprint(NaN_per_feature)

## ----------------------------------------------------------------------------
##                                  OUTLIERS
## ----------------------------------------------------------------------------

### Task 2: Remove outliers
data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
print data_dict['LOCKHART EUGENE E']
data_dict.pop('LOCKHART EUGENE E')

### Task 3: Create new feature(s)

### Create a pandas DataFrame
df = pd.DataFrame.from_records(list(data_dict.values()))
employees = pd.Series(list(data_dict.keys()))


### Set the index of df to be the employee series
df.set_index(employees, inplace = True)

### Convert strings to floats
df_new = df.apply(lambda x: pd.to_numeric(x, errors = 'coerce')).copy()

### Correlation of salary and bonus

plt.scatter(df_new['salary'], df_new['bonus'])
plt.xlabel('Salary')
plt.ylabel('Bonus')
plt.show()


## ----------------------------------------------------------------------------
##                               ADD NEW FEATURES
## ----------------------------------------------------------------------------


### Add new features

#messages_to_poi = from_this_person_to_poi/to_messages
#df_new["messages_to_poi"] = df_new["from_this_person_to_poi"]\
#/df_new["to_messages"]

#messages_from_poi = from_poi_to_this_person/from_messages
#df_new["messages_from_poi"] = df_new["from_poi_to_this_person"]\
#/df_new["from_messages"]

#bonus_to_salary_ratio = bonus/salary
df_new["bonus_to_salary"] = df_new["bonus"]/df_new["salary"]

#df_new["bonus_to_salary"] = df_new["bonus"]/df_new["salary"]

#Add the new features to features_list
#features_list.append("messages_to_poi")
#features_list.append("messages_from_poi")
features_list.append("bonus_to_salary")


### Replace NaN with 0
df_new.replace('NaN', 0, inplace = True)
df_new = df_new.replace(np.nan,'NaN', regex=True)


### check the features list
print list(df_new)

df_dict = df_new.to_dict('index')


### Store to my_dataset for easy export below.
my_dataset = df_dict


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
print features_list

## ----------------------------------------------------------------------------
##                                BASIC MODELS
## ----------------------------------------------------------------------------

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


# Basic classifier models

# Create a train-test split
features_train, features_test, labels_train, labels_test = \
cross_validation.train_test_split(features, labels, test_size = 0.3, 
                                  random_state = 42)
# Naive Bayes

print '\nNaive Bayes Classifier'

clf_gnb = GaussianNB()
clf_gnb.fit(features_train, labels_train)
pred = clf_gnb.predict(features_test)
accuracy = accuracy_score(labels_test, pred)

print '\nAccuracy'
print accuracy

label_names = ["Not POT", "POI"]

print '\nClassification Report'
print classification_report(y_true = labels_test, y_pred = pred, 
                            target_names = label_names)



# SVM

print '\nSupport Vector Machines'

clf_svm = SVC(kernel="rbf")
clf_svm.fit(features_train, labels_train)
pred = clf_svm.predict(features_test)
accuracy = accuracy_score(labels_test, pred)

print '\nAccuracy'
print accuracy

label_names = ["Not POT", "POI"]

print '\nClassification Report'
print classification_report(y_true = labels_test, y_pred = pred, 
                            target_names = label_names)


# Decision Tree
clf_dt = tree.DecisionTreeClassifier(random_state = 42)
clf_dt.fit(features_train,labels_train)
pred = clf_dt.predict(features_test)
accuracy = accuracy_score(labels_test, pred)
label_names = ["Not POT", "POI"]

print '\nClassification Report'
print classification_report(y_true = labels_test, y_pred = pred, 
                            target_names = label_names)

## ----------------------------------------------------------------------------
##                              FEATURE IMPORTANCES
## ----------------------------------------------------------------------------

#$ Print the feature importances for the decision tree
print '\n Feature Importances'
importances = clf_dt.feature_importances_
indices = np.argsort(importances)[::-1]
n = len(features_list)
for i in range (n-1):
    print "{} feature {} ({})".format(i+1, features_list[indices[i]+1], 
           importances[indices[i]])

## ----------------------------------------------------------------------------
##                              KBEST FEATURES
## ----------------------------------------------------------------------------

# Decision Tree classifier function
def dt_classifier(features_train, features_test, labels_train, labels_test):

    clf_dt = tree.DecisionTreeClassifier(random_state = 42)
    clf_dt.fit(features_train,labels_train)
    pred = clf_dt.predict(features_test)
    precision = precision_score(labels_test, pred)
    recall = recall_score(labels_test, pred)
    precision_list.append(precision)
    recall_list.append(recall)

    
# Calculate preicion and recall for each K value and add them to the lists
precision_list = []
recall_list = []
k_list = range(1,len(features_list))
print k_list
#k_list = [1,2,3,4,5,6,7,8,9,10,11]
for k in k_list:
    # Create a train-test split
    features_train, features_test, labels_train, labels_test = \
    cross_validation.train_test_split(features, labels, test_size = 0.3, 
                                  random_state = 42)

    selector = SelectKBest(f_classif, k=k)
    selector.fit(features_train, labels_train)
    k_features_train = selector.transform(features_train)
    k_features_test = selector.transform(features_test)
        
    dt_classifier(k_features_train, k_features_test, labels_train, labels_test)


# Create a dataframe from the 3 lists for K, precision and recall
metric_k = zip(k_list,precision_list, recall_list)
metric_dict = dict(zip(k_list, metric_k))
metric_df = pd.DataFrame.from_records(list(metric_dict.values()), 
                                      columns = ["K", "Precision", "Recall"])
#print metric_df

# Plot the k value against precision and recall
import matplotlib.pyplot as plt
plt.plot(metric_df.K,metric_df.Precision, marker ='o')
plt.plot(metric_df.K,metric_df.Recall, marker = 'X')
plt.xlabel('K best features')
plt.ylabel('Score')
plt.title('K best features vs. Precision and Recall score')
plt.legend(['Precision', 'Recall'])
#plt.savefig('K_Vs_Scores.png')
plt.show()


# Calculate the K-score for all features, rank them in descending order
   
selector = SelectKBest(f_classif,k='all')
selector.fit(features_train, labels_train)
support = selector.get_support(indices = True)
k_best_features = {}
for i, score in enumerate(selector.scores_):
    feature = features_list[support[i] + 1]
    k_best_features[feature] = score
    #print(features_list[support[i] + 1], score)
k_best_features.pop('email_address')

k_best_df = pd.DataFrame(k_best_features.items(), columns = ["feature", "score"])
k_best_df = k_best_df.sort_values('score', ascending = False)

# Plot the K-scores in a bar chart

import seaborn as sns
fig,ax = plt.subplots(figsize =(8,8))
ax.figure.subplots_adjust(bottom = 0.3)
ax = sns.barplot(data = k_best_df, x = "feature", y = "score",  color = "blue")
ax.set_xlabel("Features")
ax.set_title("SelectKBest scores for features")
for item in ax.get_xticklabels():
    item.set_rotation(90)
#fig = ax.get_figure()   
#fig.savefig('kbest_bar.png')


## ----------------------------------------------------------------------------
##                          IMPROVING THE MODELS
## ---------------------------------------------------------------------------- 

# Implement feature scaling, selectkbest and PCA on the 3 classifiers to see
# which ones perform best

scaler = StandardScaler()
kbest = SelectKBest()
pca = PCA()
nb = GaussianNB()
dt = tree.DecisionTreeClassifier(random_state=42)
sv = SVC(kernel ='rbf', C = 1000)



sss = StratifiedShuffleSplit(n_splits = 10, 
                             test_size = 0.3, 
                             random_state = 180)


# Create a combined feature of SlectKBest and PCA

combined_features = FeatureUnion([("reduce_dim", pca), ("selection", kbest)])

 
# Create a pipeline - NB
        
pipe_nb = Pipeline(steps = [("scaling", scaler), ("features", combined_features),
                         ("classifier", nb)])
    

# List of parameters for GridSearchCV
params ={'features__selection__k' : [6],
         'features__reduce_dim__n_components' : range(1,len(features_list))
         }
 
# Naive Bayes Classifer
   
clf_nb = GridSearchCV(pipe_nb,
                      param_grid = params,
                      n_jobs=1, 
                      cv=sss, 
                      scoring='f1')

clf_nb.fit(features, labels)


# Decision Tree

# Create a pipeline - DT
pipe_dt = Pipeline(steps = [("scaling", scaler), ("features", combined_features),
                         ("classifier", dt)])

# Decision Tree classifier    
clf_dt = GridSearchCV(pipe_dt,
                      param_grid = params,
                      n_jobs=1, 
                      cv=sss, 
                      scoring='f1')

clf_dt.fit(features, labels)



# SVM
# Create a pipeline - SVM
pipe_sv = Pipeline(steps = [("scaling", scaler), ("features", combined_features),
                         ("classifier", sv)])


#SVM Classifier    
clf_sv = GridSearchCV(pipe_sv,
                      param_grid = params,
                      n_jobs=1, 
                      cv=sss, 
                      scoring='f1')
clf_sv.fit(features, labels)

# Pass the best classifers from Naive Bayes, Decision Tree and SVM
# GridSearchCV to the test classifier to evaluate the performance on test data
# set

test_classifier(clf_nb.best_estimator_, my_dataset, features_list)
test_classifier(clf_dt.best_estimator_, my_dataset, features_list)
test_classifier(clf_sv.best_estimator_, my_dataset, features_list)

## ----------------------------------------------------------------------------
##                          PICK DECISION TREE FOR TUNING
## ----------------------------------------------------------------------------


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.
### StratifiedShuffleSplit.html




# Define the scaler,PCA, SelectKBest and Classifier variables
scale = StandardScaler()
pca = PCA()
kbest = SelectKBest()
DT = tree.DecisionTreeClassifier(random_state = 42)

# Create a combined feature of SlectKBest and PCA
combined_features = FeatureUnion([("reduce_dim", pca), ("selection", kbest)])


# Create a pipeline
pipe = Pipeline(steps = [("scaler", scale), ("features", combined_features),
                         ("classifier", DT)])
    


# Define the parameters
params = {
        'features__selection__k' : [6],
        'features__reduce_dim__n_components' : [2],
        'classifier__criterion' : ['gini', 'entropy'],
        'classifier__max_leaf_nodes' : [None,5,10,20],
        'classifier__max_depth' : [4,5,6,7],
        'classifier__min_samples_leaf' : [1,5,10],
        'classifier__min_samples_split' : [2,4,6]
        }


# Create a StratifiedShuffleSplit to train the data on 100 splits (this is
# useful as our dataset is sparse)
sss = StratifiedShuffleSplit(n_splits = 100, 
                             test_size = 0.3, 
                             random_state = 60)

# Build and fit the classifier
clf = GridSearchCV(pipe, 
                   param_grid = params, 
                   n_jobs=1, 
                   cv=sss, 
                   scoring='f1')

clf.fit(features,labels)


# Print out the best parameters, features selected
features_k = clf.best_params_['features__selection__k']
SKB_k=SelectKBest(f_classif, k=features_k)
SKB_k.fit_transform(features,labels)   
features_selected=[features_list[1:][i]for i in SKB_k.get_support(indices=True)]
print features_selected

    
print "\n Best Parameters"
print clf.best_params_


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

best_clf = clf.best_estimator_
dump_classifier_and_data(best_clf, my_dataset, features_list)

#test_classifier(clf.best_estimator_, my_dataset, features_list)