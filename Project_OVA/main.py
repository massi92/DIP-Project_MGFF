import numpy as np
import cv2
import os
import sys
import getopt

from FeaturesFile import FeaturesFile
from FeaturesLearning import FeaturesLearning
from FileManager import FileManager
from detectAndExtract import detectAndExtract


n_folds = input("Insert the number of folds: ")

detectors_descriptors=[["MSER","SIFT"],["HARRIS","SIFT"],["SIFT","SIFT"],["ORB","ORB"],["FAST","SURF"],["FAST","BRIEF"]]

fm = FileManager()

categories = fm.listNoHiddenDir(os.path.dirname(__file__)+os.path.sep+"\\Project_OVA\\imm")

for i in range(0,len(categories)):

	currentCategories = categories[i]
	
	print "\n######################################"
	print "CURRENT CATEGORY: " + currentCategories		
	

	for x in range(0,len(detectors_descriptors)):		 
		
		print "###########################"
		
		print currentCategories
		
		print "######################################"
		print "Features: " + detectors_descriptors[x][0] + " - " + detectors_descriptors[x][1]
		ff = FeaturesFile(detectors_descriptors[x][0],detectors_descriptors[x][1],currentCategories)
		X_positive, X_negative = ff.getFeatures(currentCategories)
				
		fl = FeaturesLearning(X_positive,X_negative,ff)
		fl.trainModel(n_folds)	