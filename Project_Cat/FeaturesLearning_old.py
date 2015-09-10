import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cross_validation import cross_val_score
from sklearn import decomposition
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
    
    clf = RandomForestClassifier(n_estimators=300,verbose=False,n_jobs=-1)
    #clf = AdaBoostClassifier(n_estimators=300)
  
    def __init__(self, X_positive, X_negative,ffe):
        super(FeaturesLearning, self).__init__()
        self.X_positive = X_positive
        self.X_negative = X_negative
        self.X_total = X_positive + X_negative
        # 0 positivo, 1 negativo
        self.ff = ffe
        self.y_total = [0]*len(X_positive)+[1]*len(X_negative) 
    
    def trainModel(self,path,n_folds):
        kf = cross_validation.KFold(len(self.X_total),n_folds=n_folds,shuffle=True,random_state=random.randint(1,10))
        #print kf
        k = 0
        cross_val_scores = []
        test_scores = []
        predictions = []
        probabilities = []
        for (train_index,test_index) in (kf):
            #print "a"
            """
            print train_index
            print "#################"
            print test_index
            print "############"
            raw_input()
            """
            self.X_train = [self.X_total[i] for i in train_index]
            self.X_test = [self.X_total[i] for i in test_index]
            self.y_train = [self.y_total[i] for i in train_index]
            self.y_test = [self.y_total[i] for i in test_index]

            self.clf = self.clf.fit(self.X_train,self.y_train)

            scores = cross_val_score(self.clf,self.X_train,self.y_train)

            cross_val_scores = cross_val_scores + [scores.mean()]
            test_scores = test_scores + [self.clf.score(self.X_test,self.y_test)]
            y_pred, prob = self._prediction(self.ff,path)
            predictions = predictions + [y_pred[0]]
            probabilities = probabilities + [prob[0]] #probabilita che appartenga alla classe positiva - 0
            print "Probabilita' " + str(k) + " -> " + str(prob)
            k = k+1

        print "Indici di prestazioni (medie)"
        print "cross_val_score: " + str(np.array(cross_val_scores).mean())
        print "test score: " + str(np.array(test_scores).mean())                  
        print "predictions: " + str(predictions)
        #print "probabilities: " + self.ff.getKeyword() + str(probabilities)
        return [self.ff.getKeyword(),probabilities]

    def _prediction(self,featuresFile,path,class_test=None):
        obj = featuresFile.getObj()
        X_test = [obj.elabora(path)]

        # Previsione
        y_pred = self.clf.predict(X_test)

        # Da ritornare vettore probabilita
        prob_prev = self.clf.predict_proba(X_test)
        #print prob_prev
        #print "La classe prevista e': " + str(y_pred)
        #if class_test != None:
           # print " e quella corretta e': " + str(class_test)

        return y_pred,np.array(prob_prev).ravel()

