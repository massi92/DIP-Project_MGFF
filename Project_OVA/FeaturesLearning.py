import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
import math

from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report,precision_score,recall_score,accuracy_score
from sklearn import decomposition
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation

from FeaturesFile import FeaturesFile
from FileManager import FileManager
from detectAndExtract import detectAndExtract
from HOG import HOG

class FeaturesLearning(object):
    """docstring for FeaturesLearning"""

    X_total = []
    y_total = []
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    
    X_positive = None
    X_negative = None

    ff = None

    pca = None
    
    clf = RandomForestClassifier(n_estimators=300,verbose=False,n_jobs=-1,criterion="entropy")
  
    def __init__(self, X_positive, X_negative,ffe):
        super(FeaturesLearning, self).__init__()
        self.X_positive = X_positive
        self.X_negative = X_negative
        random.shuffle(self.X_negative)
        self.X_total = self.X_positive + self.X_negative

        # 0 positivo, 1 negativo
        self.ff = ffe
        self.y_total = [0]*len(X_positive)+[1]*len(X_negative) 
    
    def trainModel(self,folds):
        
        kf = cross_validation.StratifiedKFold(self.y_total,n_folds=folds,shuffle=True,random_state=random.randint(1,100))      

        num_X_positive = len(self.X_positive)
        num_testing = math.ceil(float(num_X_positive)/folds)
        num_training = (num_X_positive-num_testing)

        num_testing = int(num_testing * 2)
        num_training = int(num_training * 2)


        for (train_index,test_index) in (kf):

            self.X_train = [self.X_total[i] for i in train_index] [0:num_training]
            self.X_test = [self.X_total[i] for i in test_index] [0:num_testing]
            self.y_train = [self.y_total[i] for i in train_index] [0:num_training]
            self.y_test = [self.y_total[i] for i in test_index] [0:num_testing]
            
            print "################"
            print "Original"
            print np.array(self.y_test)
            print "################"
            self.clf = self.clf.fit(self.X_train,self.y_train)
            print "Predicted"
            y_pred = self.clf.predict(self.X_test)
            print y_pred
            print "################"
            print "Evaluation\n"
            cm = confusion_matrix(self.y_test,y_pred)
            print cm
            print "Precision Score:"
            print precision_score(self.y_test,y_pred,average="binary")
            print "Recall Score:"
            print recall_score(self.y_test,y_pred,average="binary") 
            print "Accuracy Score:"
            print accuracy_score(self.y_test,y_pred)