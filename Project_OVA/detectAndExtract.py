import numpy as np
import cv2
import os
from preprocess import Preprocess

"""
Questa Classe contiene il metodo "elabora" che estrae dei detector e da tali detector estrai descriptor per le features delle immagini.
"""

"""
Supported Feature Detector Algorithms
FAST FastFeatureDetector
STAR StarFeatureDetector
SIFT SIFT (nonfree module)
SURF SURF (nonfree module)
ORB ORB
BRISK BRISK
MSER MSER
GFTT GoodFeaturesToTrackDetector
HARRIS GoodFeaturesToTrackDetector with Harris detector enabled
Dense DenseFeatureDetector
SimpleBlob SimpleBlobDetector


Supported Descriptor Extractor Algorithms

SIFT SIFT
SURF SURF
BRIEF BriefDescriptorExtractor
BRISK BRISK
ORB ORB
FREAK FREAK       

"""
class detectAndExtract:
	_detector = None
	_descriptor = None

	def __init__(self,detector,descriptor):
		#super(detectAndExtract, self).__init__()
		self._detector = detector
		self._descriptor = descriptor
			
	def elabora(self,path):
		img1 = cv2.imread(path,0)  	
		detectorObj = cv2.FeatureDetector_create(self._detector)
		#detectorObj.setInt("nfeatures",1000)
		
		descriptorExtractor = cv2.DescriptorExtractor_create(self._descriptor)
		
		keypoints = detectorObj.detect(img1)
		
		(keypoints, descriptors) = descriptorExtractor.compute(img1, keypoints)
		  
		out = descriptors[0]

		for i in range(1,len(descriptors)):
			np.concatenate([out,descriptors[i]])	
		
		return out.tolist()