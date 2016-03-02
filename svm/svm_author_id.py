#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
"""
# Just normal linear kernel

(venv) Kyumins-MacBook-Pro:svm math4tots$ time python svm_author_id.py 
no. of Chris training emails: 7936
no. of Sara training emails: 7884
[LibSVM]..
Warning: using -h 0 may be faster
*.*
optimization finished, #iter = 3455
obj = -1250.730984, rho = 0.988532
nSV = 2438, nBSV = 1565
Total nSV = 2438
score =  0.984072810011

real	3m2.275s
user	3m0.666s
sys	0m1.088s

----------------------------------------------------------

# linear kernel with 1% training set.

(venv) Kyumins-MacBook-Pro:svm math4tots$ time python svm_author_id.py 
no. of Chris training emails: 7936
no. of Sara training emails: 7884
[LibSVM]*.*
optimization finished, #iter = 164
obj = -67.539550, rho = 0.507895
nSV = 142, nBSV = 72
Total nSV = 142
score =  0.884527872582

real	0m5.827s
user	0m5.438s
sys	0m0.350s

----------------------------------------------------------

# rbf kernel with 1% training set.

(venv) Kyumins-MacBook-Pro:svm math4tots$ time python svm_author_id.py 
no. of Chris training emails: 7936
no. of Sara training emails: 7884
[LibSVM]*
optimization finished, #iter = 79
obj = -157.929803, rho = -0.001091
nSV = 158, nBSV = 158
Total nSV = 158
score =  0.616040955631

real	0m5.907s
user	0m5.556s
sys	0m0.320s

----------------------------------------------------------

# rbf kernel, C=10000, full training data.

(venv) Kyumins-MacBook-Pro:svm math4tots$ time python svm_author_id.py 
no. of Chris training emails: 7936
no. of Sara training emails: 7884
[LibSVM]...*...*
optimization finished, #iter = 6027
obj = -4521394.113914, rho = -68.770266
nSV = 1539, nBSV = 411
Total nSV = 1539
score =  0.990898748578

real	1m49.848s
user	1m49.192s
sys	0m0.575s

"""

USE_ONE_PERCENT = False

## Later part -- try out on 1% data set.
if USE_ONE_PERCENT:
  features_train = features_train[:len(features_train)/100] 
  labels_train = labels_train[:len(labels_train)/100] 

from sklearn.svm import SVC
clf = SVC(kernel='rbf', verbose=3, C=10000)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
from sklearn.metrics import accuracy_score
score = accuracy_score(labels_test, pred)
print "score = ", score
print 'pred[10] = ', pred[10]
print 'pred[26] = ', pred[26]
print 'pred[50] = ', pred[50]

print 'total Chris = ', sum(1 for p in pred if p == 1)

#########################################################


