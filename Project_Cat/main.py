import numpy as np
import cv2
import os
import sys

from FeaturesFile import FeaturesFile
from FeaturesLearning import FeaturesLearning
from FileManager import FileManager
from detectAndExtract import detectAndExtract


n_folds = input("Insert the number of folds: ")

""" COSTANTI E VARIABILI """
detectors_descriptors=[["MSER","SIFT"],["HARRIS","SIFT"],["SIFT","SIFT"],["ORB","ORB"],["FAST","SURF"],["FAST","BRIEF"]]

for i in range(0,len(detectors_descriptors)):
	print "\n######################################"
	name = detectors_descriptors[i][0] + "-" + detectors_descriptors[i][1]
	print name
	ff = FeaturesFile(detectors_descriptors[i][0],detectors_descriptors[i][1])
	ff.getFeatures()
	X_train, y_train,_ = ff.featuresCategories()
	print "#####################"
	fl = FeaturesLearning(X_train,y_train,ff)
	fl.trainModel(n_folds)
